"""kvlab/warm_disk.py on CPU: the candidate planner against the real restore path, and the preparation
procedure. Uses a tiny random bf16 Qwen2 with C = 4 and real host/disk tiers (pin=False).

The planner's expectations are not its own output:
- every planner result is compared with the disk reads the unchanged engine actually attempts
  (`AlwaysRestoreEngine.prefill_request` -> `TieredKVCache.restore` -> `DiskStore.read_into`, recorded by a
  subclass);
- the hand-placed layouts also state the expected chunk indices literally.

The preparation tests drive the real plan with scripted page observations and reads, so they never depend
on real memory reclaim."""

import builtins
import errno
import json
import os
import random
import time

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import traces as tr
from kvlab import warm_disk as wd
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.tiered_cache import TieredKVCache

C = 4


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)


class RecordingDisk(DiskStore):
    """The unchanged DiskStore, recording every read the restore attempts (in order). `fail` maps a key to an
    OSError raised by the controlled I/O boundary (readinto_fn) while that key's file is being read."""

    def __init__(self, *a, **k):
        self.attempts, self.fail, self._current = [], {}, None
        super().__init__(*a, readinto_fn=self._readinto, **k)

    def _readinto(self, f, mv):
        if self._current in self.fail:
            raise self.fail[self._current]
        return cf.default_readinto(f, mv)

    def read_into(self, key, ns, ident, dest):
        self.attempts.append(key)
        self._current = key
        try:
            return super().read_into(key, ns, ident, dest)
        finally:
            self._current = None


def make(ns, tmp_path, host_slots=8, disk_files=16, staging=True, name="disk"):
    cb = cf.payload_len(ns)
    budget = (host_slots + 1) * cb if staging else cb - 1  # below one chunk: no staging buffer and no slots
    host = HostStore(budget, cb, max_entries=host_slots, pin=False)
    disk = RecordingDisk(tmp_path / name, disk_files * cf.file_size(ns), max_entries=disk_files)
    return TieredKVCache(ns, host, disk)


def payload(ns, seed) -> bytes:
    g = torch.Generator().manual_seed(seed)
    n = cf.payload_len(ns) // 2
    return torch.randn(n, generator=g).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()


def place(t, toks, layout: str):
    """layout[i] for chunk i: 'h' host, 'd' disk, 'b' both (host entry plus a disk file), '-' absent."""
    keys = chain_keys(t.ns_digest, toks, C)
    for i, c in enumerate(layout):
        ident = ChunkIdentity.for_chunk(t.ns_digest, toks, C, i, keys)
        if c in "db":
            ok, why = t.disk.put(keys[i], cf.encode_header(t.ns, ident), bytearray(payload(t.ns, i)))
            assert ok, why
        if c in "hb":
            slot, why = t.host.allocate(t._spill)
            assert why == "ok"
            t.host.slot_view(slot).copy_(torch.frombuffer(bytearray(payload(t.ns, 100 + i)), dtype=torch.uint8))
            t.host.insert(keys[i], ident, slot)
    return keys


def toks(n, start=100):
    return list(range(start, start + n))


def run(model, t, prompt):
    """Plan, then the real request. Returns (plan, read attempts, Prefilled)."""
    plan = wd.plan_disk_reads(t, prompt)
    t.disk.attempts.clear()
    p = AlwaysRestoreEngine(model, t).prefill_request(prompt)
    return plan, list(t.disk.attempts), p


def idx(plan):
    return [i for i, _k in plan["candidates"]]


@pytest.mark.parametrize("n_tokens, want", [(4, []), (5, [0]), (8, [0]), (9, [0, 1]), (12, [0, 1]), (13, [0, 1, 2])])
def test_exact_chunk_boundaries_follow_the_first_token_rule(model, ns, tmp_path, n_tokens, want):
    """Every complete chunk of the prompt is on disk; only L <= N - 1 on the chunk grid is eligible."""
    t = make(ns, tmp_path)
    prompt = toks(n_tokens)
    keys = place(t, prompt, "d" * (n_tokens // C))
    plan, reads, p = run(model, t, prompt)
    assert idx(plan) == want and plan["eligible_chunks"] == len(want)
    assert reads == [keys[i] for i in want]  # the files the engine actually read, in order
    assert p.restore_len == len(want) * C and p.sources == ["disk"] * len(want)


@pytest.mark.parametrize("layout, n_tokens, want, stop, sources", [
    ("", 13, [], "first_absent", []),                      # empty cache
    ("-dd", 13, [], "first_absent", []),                   # first chunk absent: later disk files are not read
    ("dd-d", 17, [0, 1], "first_absent", ["disk", "disk"]),  # a hole stops the prefix
    ("hdhd", 17, [1, 3], "all_eligible_present", ["host", "disk", "host", "disk"]),  # mixed tiers
    ("hhh", 13, [], "all_eligible_present", ["host"] * 3),  # host only: nothing to read
    ("bd", 9, [1], "all_eligible_present", ["host", "disk"]),  # host takes precedence over a disk copy
])
def test_candidates_equal_the_actual_reads_on_the_normal_path(model, ns, tmp_path, layout, n_tokens, want, stop,
                                                                sources):
    t = make(ns, tmp_path)
    prompt = toks(n_tokens)
    keys = place(t, prompt, layout)
    plan, reads, p = run(model, t, prompt)
    assert idx(plan) == want and plan["stop"] == stop
    assert reads == [keys[i] for i in want]
    assert p.sources == sources and plan["host_chunks"] == sources.count("host")
    assert plan["present_chunks"] == len(sources)


def test_no_staging_buffer_means_no_candidate_and_no_read(model, ns, tmp_path):
    t = make(ns, tmp_path, staging=False)
    assert t.host.staging_slots == 0
    prompt = toks(13)
    place(t, prompt, "ddd")
    plan, reads, p = run(model, t, prompt)
    assert plan["candidates"] == [] and plan["stop"] == "no_staging_buffer" and plan["present_chunks"] == 3
    assert reads == [] and p.restore_len == 0 and p.miss_reason == "no_staging_buffer"


def _corrupt(t, key):
    path = t.disk.root / f"{key.hex()}.kvc"
    raw = bytearray(path.read_bytes())
    raw[-6] ^= 0x40  # a payload byte: the payload CRC fails after the whole file is read
    path.write_bytes(raw)


@pytest.mark.parametrize("fault, reason", [("payload_crc", "disk_payload_crc"), ("eio", "disk_read_oserror:EIO"),
                                           ("missing", "disk_file_missing")])
def test_invalid_or_failed_file_gives_a_conservative_superset(model, ns, tmp_path, fault, reason):
    """The restore attempts the bad file and stops. The candidates contain every attempted file (a prefix of
    them) and possibly later files that were never read."""
    t = make(ns, tmp_path)
    prompt = toks(17)
    keys = place(t, prompt, "dddd")
    if fault == "payload_crc":
        _corrupt(t, keys[1])
    elif fault == "eio":
        t.disk.fail[keys[1]] = OSError(errno.EIO, "injected I/O error")
    else:
        os.unlink(t.disk.root / f"{keys[1].hex()}.kvc")
    plan, reads, p = run(model, t, prompt)
    assert idx(plan) == [0, 1, 2, 3]
    assert reads == [keys[0], keys[1]]                      # attempted: the valid file, then the bad one
    assert reads == [k for _i, k in plan["candidates"]][:len(reads)] and set(reads) < {k for _i, k in plan["candidates"]}
    assert p.restore_len == C and p.miss_reason == reason


def _snapshot(t):
    h, d = t.host, t.disk
    files = {e.name: (e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(d.root)}
    return json.dumps({"host": [(k.hex(), e.slot, e.pins) for k, e in h.entries.items()], "free": list(h._free),
                       "staging_busy": h._staging_busy, "host_evictions": h.evictions,
                       "disk": [(k.hex(), v) for k, v in d.entries.items()], "undeletable": dict(d.undeletable),
                       "disk_bytes": (d.committed_bytes, d.tmp_bytes, d.peak_bytes, d.evictions, d.stale_removed),
                       "counters": t.counters.to_dict(), "last_restore": t.last_restore, "files": files,
                       "attempts": [k.hex() for k in d.attempts]}, sort_keys=True)


def test_planning_has_no_side_effect_and_opens_no_file(model, ns, tmp_path, monkeypatch):
    t = make(ns, tmp_path)
    prompt = toks(25)
    place(t, prompt, "hdbd-d")
    _corrupt(t, chain_keys(t.ns_digest, prompt, C)[3])  # a bad file must not be detected, repaired or removed
    before = _snapshot(t)

    def refuse(*a, **k):
        raise AssertionError("the planner opened a file")
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", refuse)
        m.setattr(os, "open", refuse)
        plan = wd.plan_disk_reads(t, prompt)
        plan2 = wd.plan_disk_reads(t, prompt)
    assert _snapshot(t) == before and plan == plan2
    assert idx(plan) == [1, 3] and plan["stop"] == "first_absent"  # 'b' at index 2: host precedence


def _trace_prompts():
    gens = [tr.GENERATORS["doc_reuse"](7, vocab=(10, 500), n_requests=10, n_docs=3, instruction_len=4, doc_min=4,
                                       doc_max=11, question_min=1, question_max=3, zipf_s=1.0),
            tr.GENERATORS["multi_turn"](8, vocab=(10, 500), n_conversations=2, n_turns=4, opening_len=6, user_min=1,
                                        user_max=4, answer_min=1, answer_max=4)]
    prompts = [list(r.tokens) for g in gens for r in g.requests]
    random.Random(0).shuffle(prompts)
    return prompts


def test_trace_replay_candidates_equal_reads_and_planning_changes_nothing(model, ns, tmp_path):
    """Two fresh instances replay the same prompts with spills and disk evictions (2 host slots, 3 disk files).
    One plans before every request, the other never plans. Every plan equals the reads attempted next; both
    instances make identical decisions and end in identical states."""
    a, b = make(ns, tmp_path, host_slots=2, disk_files=3, name="a"), make(ns, tmp_path, host_slots=2, disk_files=3,
                                                                          name="b")
    seen = {"disk": 0, "mixed": 0, "empty_with_disk_entries": 0}
    for prompt in _trace_prompts():
        plan, reads, pa = run(model, a, prompt)
        b.disk.attempts.clear()
        pb = AlwaysRestoreEngine(model, b).prefill_request(prompt)
        assert reads == [k for _i, k in plan["candidates"]]
        assert (pa.restore_len, pa.sources, pa.stored_chunks, pa.miss_reason) == \
            (pb.restore_len, pb.sources, pb.stored_chunks, pb.miss_reason)
        assert [k.hex() for k in b.disk.attempts] == [k.hex() for k in reads]
        seen["disk"] += bool(reads)
        seen["mixed"] += "host" in pa.sources and "disk" in pa.sources
        seen["empty_with_disk_entries"] += not reads and len(a.disk) > 0
    sa, sb = json.loads(_snapshot(a)), json.loads(_snapshot(b))
    for s in (sa, sb):
        s.pop("files")
    assert sa == sb
    assert a.counters.disk_hits > 0 and a.disk.evictions > 0 and a.counters.host_spills_to_disk > 0
    assert all(v > 0 for v in seen.values()), seen  # the replay exercised every situation


# ---- the preparation procedure, with scripted observations and reads

class Script:
    """Scripted page observations per (file, pass) and reads; logs every call in order."""

    def __init__(self, t, pages=3, size=3 * 4096 - 7):
        self.t, self.pages, self.size, self.log = t, pages, size, []
        self.obs, self.reads, self.n_obs = {}, {}, {}

    def observe(self, path):
        k = self.n_obs[path.name] = self.n_obs.get(path.name, 0) + 1
        self.log.append(("observe", path.name))
        v = self.obs.get((path.name, k), self.pages)
        if isinstance(v, BaseException):
            raise v
        return v, self.pages, self.size

    def read(self, path):
        self.log.append(("read", path.name))
        v = self.reads.get(path.name, self.size)
        if isinstance(v, BaseException):
            raise v
        return v

    def counters(self):
        self.log.append(("counters",))
        return {"status": "ok", "vmstat": {"drop_pagecache": 5}}

    def prepare(self, prompt):
        """prepare(), then complete() as the runner does after its timed region (t0 = the end of the final pass)."""
        rec = wd.prepare(self.t, prompt, observe=self.observe, read=self.read, read_counters=self.counters)
        return wd.complete(rec, rec["t_verified_ns"], {"status": "ok", "vmstat": {"drop_pagecache": 5}})


def _setup(ns, tmp_path, layout="dhd", n_tokens=13):
    t = make(ns, tmp_path)
    prompt = toks(n_tokens)
    keys = place(t, prompt, layout)
    names = [f"{k.hex()}.kvc" for k in keys]
    return t, prompt, names


def test_order_is_plan_initial_pass_one_prewarm_final_pass(ns, tmp_path):
    t, prompt, names = _setup(ns, tmp_path)
    a, c = names[0], names[2]  # chunk 1 is in the host: not a candidate
    s = Script(t)
    s.obs[(c, 1)] = 1  # 1 of 3 pages initially
    rec = s.prepare(prompt)
    assert s.log == [("counters",), ("observe", a), ("observe", c), ("read", c), ("observe", a), ("observe", c)]
    assert rec["candidates"] == [a, c] and [f["index"] for f in rec["files"]] == [0, 2]
    fa, fc = rec["files"]
    assert fa["prewarm"] == {"status": "not_needed"} and fc["prewarm"]["status"] == "read"
    assert fc["prewarm"]["bytes_read"] == s.size and fc["initial"]["resident_pages"] == 1
    assert fc["final"]["resident_pages"] == 3 and rec["prewarmed_files"] == 1
    assert rec["phases"] == {"initial": "done", "prewarm": "done", "final": "done"} and rec["outcome"] == "verified"
    assert rec["plan"] == {"eligible_chunks": 3, "present_chunks": 3, "host_chunks": 1, "stop": "all_eligible_present"}
    assert rec["t_start_ns"] <= rec["t_planned_ns"] <= fa["initial"]["t_ns"] <= fc["initial"]["t_ns"] \
        <= fc["prewarm"]["t_start_ns"] <= fc["prewarm"]["t_end_ns"] <= fa["final"]["t_ns"] <= fc["final"]["t_ns"] \
        <= rec["t_verified_ns"]
    assert rec["counters_before"]["vmstat"]["drop_pagecache"] == 5 and rec["tier_entries"] == 2


def test_only_the_closing_clock_read_follows_the_final_pass(ns, tmp_path, monkeypatch):
    """Review P2-2: the outcome, phase labels and counts are computed after the timed region (complete()), not
    between the final pass and the timer."""
    t, prompt, names = _setup(ns, tmp_path)
    s = Script(t)
    s.obs[(names[2], 1)] = 1
    clock_log = s.log

    def clock():
        clock_log.append(("clock",))
        return time.perf_counter_ns()

    def no_outcome(files):
        raise AssertionError("outcome computed inside prepare()")
    real_outcome = wd.outcome
    monkeypatch.setattr(wd, "outcome", no_outcome)
    rec = wd.prepare(t, prompt, observe=s.observe, read=s.read, read_counters=s.counters, clock=clock)
    # the last final observation, its own timestamp, then the closing read: nothing else
    assert s.log[-3:] == [("observe", names[2]), ("clock",), ("clock",)]
    assert rec["files"][-1]["final"]["t_ns"] <= rec["t_verified_ns"]
    assert not {"outcome", "phases", "prewarmed_files", "timer_start_minus_verified_ns",
                "counters_after_request"} & set(rec)
    monkeypatch.setattr(wd, "outcome", real_outcome)
    out = wd.complete(rec, rec["t_verified_ns"] + 7, {"status": "ok"})
    assert out is rec and rec["outcome"] == "verified" and rec["prewarmed_files"] == 1
    assert rec["timer_start_minus_verified_ns"] == 7 and rec["counters_after_request"] == {"status": "ok"}


def test_final_pass_runs_even_when_nothing_needs_a_prewarm(ns, tmp_path):
    t, prompt, names = _setup(ns, tmp_path)
    s = Script(t)
    rec = s.prepare(prompt)
    assert [x[0] for x in s.log] == ["counters", "observe", "observe", "observe", "observe"]
    assert rec["phases"]["prewarm"] == "not_needed" and rec["outcome"] == "verified" and rec["prewarmed_files"] == 0


def test_still_not_resident_after_one_prewarm_is_recorded_without_retry(ns, tmp_path):
    t, prompt, names = _setup(ns, tmp_path)
    c = names[2]
    s = Script(t)
    s.obs[(c, 1)], s.obs[(c, 2)] = 0, 2  # after its one read, 2 of 3 pages
    rec = s.prepare(prompt)
    assert s.log.count(("read", c)) == 1 and len([x for x in s.log if x[0] == "read"]) == 1
    assert rec["outcome"] == "not_resident" and rec["files"][1]["final"]["resident_pages"] == 2


def test_a_file_resident_initially_but_lost_by_the_final_pass_is_not_resident(ns, tmp_path):
    t, prompt, names = _setup(ns, tmp_path)
    s = Script(t)
    s.obs[(names[0], 2)] = 2
    rec = s.prepare(prompt)
    assert not any(x[0] == "read" for x in s.log) and rec["outcome"] == "not_resident"


@pytest.mark.parametrize("case", ["initial observation", "final observation", "prewarm read", "short read",
                                  "empty file"])
def test_preparation_errors_are_recorded_and_give_outcome_error(ns, tmp_path, case):
    t, prompt, names = _setup(ns, tmp_path)
    c = names[2]
    s = Script(t)
    if case == "initial observation":
        s.obs[(c, 1)] = FileNotFoundError(errno.ENOENT, "gone")
    elif case == "final observation":
        s.obs[(c, 2)] = OSError(errno.EIO, "mincore")
    elif case == "prewarm read":
        s.obs[(c, 1)] = 0
        s.reads[c] = OSError(errno.EIO, "read")
    elif case == "short read":
        s.obs[(c, 1)] = 0
        s.reads[c] = s.size - 1
    else:
        s.observe = lambda path: (s.log.append(("observe", path.name)), (0, 0, 0))[1]
    rec = s.prepare(prompt)
    assert rec["outcome"] == "error", case
    f = rec["files"][1]
    if case == "initial observation":
        assert f["initial"]["status"] == "error" and "ENOENT" in f["initial"]["error"]
        assert f["prewarm"] == {"status": "skipped", "reason": "initial observation failed"}
        assert f["final"]["status"] == "observed"  # the final pass still runs and is recorded as observed
    elif case == "final observation":
        assert f["final"]["status"] == "error" and "EIO" in f["final"]["error"]
    elif case == "prewarm read":
        assert f["prewarm"]["status"] == "error" and rec["prewarmed_files"] == 0 and rec["phases"]["prewarm"] == "done"
    elif case == "empty file":
        assert f["initial"]["status"] == "error" and "empty file" in f["initial"]["error"]
    assert s.log.count(("read", c)) <= 1


def test_programming_errors_propagate(ns, tmp_path):
    t, prompt, _names = _setup(ns, tmp_path)
    s = Script(t)
    s.observe = lambda path: (_ for _ in ()).throw(TypeError("bug"))
    with pytest.raises(TypeError):
        s.prepare(prompt)


def test_empty_candidate_set_is_explicit_and_observes_nothing(ns, tmp_path):
    t, prompt, _names = _setup(ns, tmp_path, layout="-dd")
    s = Script(t)
    rec = s.prepare(prompt)
    assert s.log == [("counters",)]
    assert rec["candidates"] == [] and rec["files"] == [] and rec["outcome"] == "empty"
    assert rec["phases"] == dict.fromkeys(("initial", "prewarm", "final"), "not_applicable")
    assert rec["tier_entries"] == 2 and rec["plan"]["stop"] == "first_absent" and rec["prewarmed_files"] == 0


def test_real_files_measurement_and_prewarm(ns, tmp_path):
    """The real primitives on real chunk files: exact page counts, consistent with the V1 measurement
    (kvlab.perf_harness.resident_fraction), and a prewarm that reads every byte."""
    from kvlab.perf_harness import resident_fraction
    t, prompt, names = _setup(ns, tmp_path)
    rec = wd.prepare(t, prompt)
    size = cf.file_size(ns)
    pages = -(-size // wd.PAGE)
    for f in rec["files"]:
        for ph in ("initial", "final"):
            assert f[ph]["status"] == "observed" and (f[ph]["pages"], f[ph]["size"]) == (pages, size)
    path = t.disk.root / names[0]
    assert wd.read_file(path) == size
    r, n, sz = wd.resident_pages(path)
    assert (n, sz) == (pages, size) and r / n == resident_fraction(path)
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert wd.resident_pages(empty) == (0, 0, 0)
    with pytest.raises(FileNotFoundError):
        wd.resident_pages(tmp_path / "missing")


def test_counters_status_ok_unavailable_and_error(tmp_path):
    vm, mi = tmp_path / "vmstat", tmp_path / "meminfo"
    vm.write_text("nr_free_pages 1\ndrop_pagecache 7\ndrop_slab 7\npgsteal_kswapd 12\n")
    mi.write_text("MemTotal: 100 kB\nMemAvailable: 40 kB\nCached: 9 kB\n")
    c = wd.counters(vm, mi)
    assert c["status"] == "ok" and c["vmstat"] == {"drop_pagecache": 7, "drop_slab": 7, "pgsteal_kswapd": 12}
    assert c["meminfo_kib"] == {"MemAvailable": 40, "Cached": 9}
    vm.write_text("nr_free_pages 1\npgsteal_kswapd 12\n")
    assert wd.counters(vm, mi)["status"] == "unavailable"
    c = wd.counters(tmp_path / "no_vmstat", tmp_path / "no_meminfo")
    assert c["status"] == "error" and "ENOENT" in c["error"] and "error" in c["meminfo_kib"]
    assert wd.counters()["status"] == "ok"  # this kernel reports drop_pagecache
