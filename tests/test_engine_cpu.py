"""AlwaysRestoreEngine on CPU with a tiny randomly initialised Qwen2 model (no download, no GPU).

M1.1 regression: disk cleanup and write failures must never abort a request or
repeat inference. Every request must make exactly STEPS forward calls
(1 prefill + STEPS-1 decode steps) and must match a resident replay bitwise
(same restore length, same stored chunks). No real permissions are changed;
os.unlink is injected through m1_toy.deny_unlink.
"""

import errno

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import chain_keys, namespace_for
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import TieredKVCache
from m1_toy import deny_unlink

C, STEPS = 4, 4
PROMPT = list(range(10, 23))  # N = 13: chunks [0,4) [4,8) [8,12), eligible L = 12


@pytest.fixture(scope="module")
def tiny_model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


class Rig:
    def __init__(self, model, root, host_slots, disk_files, write_fn=cf.default_write):
        self.model = model
        self.ns = namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)
        self.file = cf.file_size(self.ns)
        cb = cf.payload_len(self.ns)
        self.host = HostStore((host_slots + 1) * cb, cb, max_entries=16, pin=False)
        self.disk = DiskStore(root, disk_files * self.file, max_entries=16, write_fn=write_fn)
        self.tiered = TieredKVCache(self.ns, self.host, self.disk)
        self.engine = AlwaysRestoreEngine(model, self.tiered)
        self.ref = ResidentReference(model, self.ns.digest(), C)
        self.keys = chain_keys(self.ns.digest(), PROMPT, C)

    def name(self, i):
        return f"{self.keys[i].hex()}.kvc"

    def run(self):
        calls = []
        hook = self.model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = self.engine.run(PROMPT, STEPS)
        finally:
            hook.remove()
        assert len(calls) == STEPS, f"{len(calls)} forward calls: inference repeated or cut short"
        ref_tokens, ref_steps = self.ref.run(PROMPT, r.restore_len, r.stored_chunks, STEPS)
        assert r.tokens == ref_tokens and all(torch.equal(a, b) for a, b in zip(r.step_logits, ref_steps))
        d = self.disk
        assert d.tmp_bytes == 0 and d.scan_dir_bytes()[0] <= d.used_bytes <= d.budget_bytes
        return r


def test_disk_eviction_unlink_failure_neither_aborts_nor_repeats_inference(tiny_model, tmp_path, monkeypatch):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1, disk_files=1)
    attempts = deny_unlink(monkeypatch, lambda n: n == rig.name(0))
    r = rig.run()  # store: chunk 0 spilled; chunk 1's spill must evict chunk 0's file, which fails
    assert (r.restore_len, r.stored_chunks) == (0, [0, 1, 2])
    assert rig.tiered.counters.spill_failures == {"evict_oserror:EACCES": 1} and attempts == [rig.name(0)]
    assert rig.disk.undeletable == {rig.name(0): rig.file} and rig.disk.used_bytes == rig.file
    r = rig.run()  # chunk 0 is not readable any more -> full recompute, still exactly STEPS calls
    assert (r.restore_len, r.miss_reason) == (0, "absent") and attempts == [rig.name(0)]


def test_invalid_entry_cleanup_failure_falls_back_without_repeating_inference(tiny_model, tmp_path, monkeypatch):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1, disk_files=4)
    r = rig.run()
    assert (r.restore_len, r.stored_chunks) == (0, [0, 1, 2])
    assert [rig.tiered.tier_of(k) for k in rig.keys] == ["disk", "disk", "host"]
    path = rig.disk.root / rig.name(1)
    raw = bytearray(path.read_bytes())
    raw[-4 - cf.payload_len(rig.ns) // 2] ^= 0x01  # one bit in the middle of the payload (before its CRC)
    path.write_bytes(raw)
    attempts = deny_unlink(monkeypatch, lambda n: n == rig.name(1))
    r = rig.run()  # chunk 1 invalid and undeletable -> verified prefix L = 4, chunk 1 recomputed and stored
    assert (r.restore_len, r.sources, r.miss_reason) == (C, ["disk"], "disk_payload_crc")
    assert r.stored_chunks == [1] and attempts == [rig.name(1)] and path.read_bytes() == bytes(raw)
    assert rig.disk.undeletable == {rig.name(1): rig.file}
    r = rig.run()  # the fresh copy of chunk 1 is in host; the invalid file is never read
    assert (r.restore_len, r.sources) == (3 * C, ["disk", "host", "disk"])


def test_temp_cleanup_failure_after_enospc_neither_aborts_nor_repeats_inference(tiny_model, tmp_path, monkeypatch):
    def enospc(f, mv):
        f.write(mv[:16])
        raise OSError(errno.ENOSPC, "No space left on device")
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1, disk_files=4, write_fn=enospc)
    attempts = deny_unlink(monkeypatch, lambda n: ".kvc.tmp" in n)
    r = rig.run()
    assert r.stored_chunks == [0, 1, 2] and len(attempts) == 2
    assert rig.tiered.counters.spill_failures == {"write_oserror:ENOSPC": 2}
    assert rig.disk.undeletable_bytes == rig.disk.used_bytes == 2 * rig.file and len(rig.disk) == 0


def test_prefill_request_without_restore_and_phase_marks(tiny_model, tmp_path):
    """M2: allow_restore=False recomputes the whole prompt in one forward call (offline
    measurement only); marks come in order; run() keeps restoring as before."""
    rig = Rig(tiny_model, tmp_path / "d", host_slots=4, disk_files=4)
    rig.run()  # caches chunks 0..2
    calls, marks = [], []
    hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        p = rig.engine.prefill_request(PROMPT, allow_restore=False, mark=marks.append)
    finally:
        hook.remove()
    assert len(calls) == 1 and marks == ["ingest", "restore", "prefill", "store"]
    assert (p.restore_len, p.sources, p.miss_reason, p.stored_chunks) == (0, [], "restore_not_attempted", [])
    assert p.cache.get_seq_length() == len(PROMPT)
    q = rig.engine.prefill_request(PROMPT)
    assert (q.restore_len, q.sources) == (3 * C, ["host"] * 3)
    r = rig.run()
    assert (r.restore_len, r.sources) == (3 * C, ["host"] * 3)
