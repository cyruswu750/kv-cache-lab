"""Request-scoped warm-disk preparation for M4_EVAL_V2 (docs/M4_EVAL_V2_PROTOCOL.md). No torch import.

M4_EVAL_V1 checked every file of a C2 instance's disk tier before each request (`kvlab.m4_bench._residency`).
That rule stays as it was, for V1. M4_EVAL_V2 replaces it with a preparation scoped to the request:

1. **Plan** (`plan_disk_reads`): the disk files that this request's restore will read on the normal path.
   This is the *candidate* set.
2. **Initial pass:** one mincore observation of every candidate.
3. **Prewarm:** at most one whole-file read of each candidate observed below full residency. There is no
   retry.
4. **Final pass:** one mincore observation of every candidate. Only the closing clock read follows it inside
   `prepare`; the record is assembled before the passes and filled in by them.
5. The caller then starts its unchanged timed region (`kvlab.m4_bench.replay_trace`). After its end, `complete`
   adds the phase labels, the prewarm count, the outcome, the interval from the final pass to the timer start,
   and the second counter snapshot. None of that bookkeeping runs between the final pass and the timer.

**What this supports.** The only claim is: every candidate file of the request was observed fully resident
in the one verification pass immediately before the request. That is a set of point observations, one per
file, at their own times. It is NOT:
- a simultaneous snapshot;
- a guarantee of residency at timer start or during the timed read (pages are not pinned, and reclaim can
  act after each observation);
- a measure of cold-disk, storage-device or production performance.

**The planner** reproduces the lookup of `AlwaysRestoreEngine.prefill_request` and `TieredKVCache.restore`
without calling them. The steps:
- the chain keys of the prompt (`kvlab.identity.chain_keys`);
- the first-token rule (`kvlab.boundaries.max_restore_len`): only complete chunks, and at least the last
  prompt token is recomputed;
- the longest leading run of present eligible keys (`tier_of`); it stops at the first absent key;
- host precedence per key;
- the staging rule: with no staging buffer, the restore stops at the first disk chunk before reading it.

It uses only dictionary membership (`HostStore.__contains__`, `DiskStore.__contains__`) and the file-name
mapping `DiskStore._final_path`. It never leases, stages, reads, validates, repairs or moves an entry in an
LRU order, and never changes a counter, so the cache state and the restore decision are unchanged.

On the normal path (every candidate valid) the candidate list equals, in order, the files the restore reads.
When a disk file is invalid or its read fails, the restore attempts that file (it may read part or all of it)
and stops. The candidate list is then a conservative superset: it contains the attempted file and possibly
later files that were never read. It never misses a file the restore attempts. Membership cannot know a file
is invalid, and finding out would need a validated read with side effects, so that superset is the intended
behaviour, not an approximation to be repaired.

**Side effects of the preparation itself** (disclosed, not hidden):
- the observations are mincore system calls on private, untouched mappings; they read no file data;
- a prewarm is an ordinary read of the whole file through a 4 MiB user buffer. It brings the file's pages
  into the page cache and refreshes their page-cache LRU state. It can push other pages out, for example
  other files of the tier or the model weights. It also passes 7 MB through the CPU caches;
- the plan computes the prompt's chain keys (SHA-256) and looks keys up in the two dictionaries, so those
  objects may be warm in the CPU caches when the timed region recomputes and looks them up again. The timed
  region itself is unchanged: its key generation and lookup still run inside it.

Errors are recorded, never hidden: an OSError while observing or reading a candidate is stored with its
text, and the request's outcome becomes "error". The validator then treats the evidence as invalid.
Programming errors (any other exception) propagate.

**Counters.** `counters()` reads /proc/vmstat (and /proc/meminfo, for context). The `drop_pagecache` count
is the subject of the batch interference rule (docs/M4_EVAL_V2_PROTOCOL.md). The other counters are context
only: they say nothing about a particular file or read.
"""

import ctypes
import errno
import mmap
import os
import time

from kvlab.boundaries import max_restore_len
from kvlab.identity import chain_keys

PROTOCOL = "request_candidates"
INTERFERENCE_RECORD = "batch_interference.json"
INTERFERENCE_FORMAT = "m4-batch-interference-v1"
RULE_COUNTER = "drop_pagecache"
VMSTAT_KEYS = (RULE_COUNTER, "drop_slab", "pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct",
               "pgsteal_file", "workingset_refault_file", "allocstall_normal", "allocstall_movable")
MEMINFO_KEYS = ("MemAvailable", "Cached", "Active(file)", "Inactive(file)")
STOPS = ("all_eligible_present", "first_absent", "no_staging_buffer")
OUTCOMES = ("empty", "verified", "not_resident", "error")
READ_SIZE = 1 << 22  # the prewarm read size of kvlab.perf_harness.warm_page_cache (V1)

PAGE = os.sysconf("SC_PAGE_SIZE")
_libc = ctypes.CDLL("libc.so.6", use_errno=True)


# ---- the plan

def plan_disk_reads(tiered, token_ids) -> dict:
    """The disk files the restore of `token_ids` would read on the normal path, with no side effect.

    Returns:
    - `eligible_chunks`: the chunks allowed by the first-token rule;
    - `present_chunks`: the leading run of eligible keys present in either tier; the restore plans exactly
      these chunks;
    - `host_chunks`: host entries among the chunks before the stop;
    - `stop`: one of STOPS;
    - `candidates`: (chunk index, key) of each disk entry the restore would read, in restore order."""
    C = tiered.C
    token_ids = list(token_ids)
    keys = chain_keys(tiered.ns_digest, token_ids, C)
    eligible = keys[:max_restore_len(len(token_ids), C) // C]
    n = 0
    while n < len(eligible) and tiered.tier_of(eligible[n]) is not None:
        n += 1
    stop = "first_absent" if n < len(eligible) else "all_eligible_present"
    candidates, host_chunks = [], 0
    for i in range(n):
        key = eligible[i]
        if key in tiered.host:
            host_chunks += 1
            continue
        if tiered.host.staging_slots == 0:  # the restore stops here without reading the file
            stop = "no_staging_buffer"
            break
        candidates.append((i, key))
    return {"eligible_chunks": len(eligible), "present_chunks": n, "host_chunks": host_chunks, "stop": stop,
            "candidates": candidates}


# ---- measurement primitives

def resident_pages(path) -> tuple[int, int, int]:
    """(resident pages, pages, bytes) of one file.

    It uses mincore(2) on an untouched private mapping. This is the measurement of
    `kvlab.perf_harness.resident_fraction`, returned as exact counts, with the size taken from the same
    descriptor that is mapped. An empty file has no pages and is not mapped: (0, 0, 0). Raises OSError."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0, 0, 0
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)  # never written, so no page is copied or touched
    finally:
        os.close(fd)
    try:
        n = (size + PAGE - 1) // PAGE
        vec = (ctypes.c_ubyte * n)()
        buf = ctypes.c_char.from_buffer(mm)
        try:
            if _libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), vec) != 0:
                e = ctypes.get_errno()
                raise OSError(e, f"mincore failed: {os.strerror(e)}")
        finally:
            del buf
        return sum(b & 1 for b in vec), n, size
    finally:
        mm.close()


def read_file(path) -> int:
    """The prewarm: one sequential, unbuffered read of the whole file. This is the loop of
    `kvlab.perf_harness.warm_page_cache` (V1), counting the bytes read. Raises OSError."""
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(READ_SIZE)
            if not b:
                return n
            n += len(b)


def _err(e: OSError) -> str:
    name = errno.errorcode.get(e.errno, str(e.errno)) if e.errno is not None else "None"
    return f"{type(e).__name__}:{name}: {e}"


def _observe(path, observe, clock) -> dict:
    try:
        r, n, size = observe(path)
    except OSError as e:
        return {"status": "error", "error": _err(e), "t_ns": clock()}
    t = clock()
    if n == 0:
        return {"status": "error", "error": "empty file: no pages to observe", "size": size, "t_ns": t}
    return {"status": "observed", "resident_pages": r, "pages": n, "size": size, "t_ns": t}


def _prewarm(path, initial: dict, read, clock) -> dict:
    if initial["status"] != "observed":
        return {"status": "skipped", "reason": "initial observation failed"}
    if initial["resident_pages"] == initial["pages"]:
        return {"status": "not_needed"}
    t = clock()
    try:
        nbytes = read(path)
    except OSError as e:
        return {"status": "error", "error": _err(e), "t_start_ns": t, "t_end_ns": clock()}
    return {"status": "read", "bytes_read": nbytes, "t_start_ns": t, "t_end_ns": clock()}


def outcome(files: list) -> str:
    """The request's preparation outcome from its per-file phases.
    - "empty": there are no candidates;
    - "error": any observation or prewarm failed, a prewarm read a byte count other than the file size, or
      the file size changed between the passes;
    - "not_resident": no error, but a candidate was below full residency in the final pass;
    - "verified": otherwise.
    The validator recomputes this independently and never trusts the recorded value."""
    if not files:
        return "empty"
    worst = "verified"
    for f in files:
        ini, pw, fin = f["initial"], f["prewarm"], f["final"]
        if ini["status"] != "observed" or fin["status"] != "observed" or pw["status"] in ("error", "skipped") \
                or fin["size"] != ini["size"] or (pw["status"] == "read" and pw["bytes_read"] != ini["size"]):
            return "error"
        if fin["resident_pages"] < fin["pages"]:
            worst = "not_resident"
    return worst


# ---- the preparation (steps 1-4)

def prepare(tiered, token_ids, *, observe=resident_pages, read=read_file, clock=time.perf_counter_ns,
            read_counters=None) -> dict:
    """Plan, initial pass, at most one prewarm per candidate, and one final pass (steps 1-4).

    Returns the request's partial record. The record is assembled before the passes and filled in by them, so
    the final pass is followed only by the clock read that closes it (`t_verified_ns`) and the return. The
    caller completes the record with `complete()` after its timed region."""
    read_counters = read_counters or counters
    t_start = clock()
    before = read_counters()  # context, and the drop_pagecache observation; read before the plan
    plan = plan_disk_reads(tiered, token_ids)
    t_planned = clock()
    disk = tiered.disk
    paths = [disk._final_path(key) for _i, key in plan["candidates"]]
    files = [{"index": i, "file": p.name} for (i, _key), p in zip(plan["candidates"], paths)]
    rec = {"protocol": PROTOCOL, "page_size": PAGE,
           "plan": {k: plan[k] for k in ("eligible_chunks", "present_chunks", "host_chunks", "stop")},
           "candidates": [f["file"] for f in files], "files": files,
           "tier_entries": len(disk) if disk is not None else 0,
           "counters_before": before, "t_start_ns": t_start, "t_planned_ns": t_planned}
    for f, p in zip(files, paths):      # 2. initial pass
        f["initial"] = _observe(p, observe, clock)
    for f, p in zip(files, paths):      # 3. at most one prewarm per candidate, no retry
        f["prewarm"] = _prewarm(p, f["initial"], read, clock)
    for f, p in zip(files, paths):      # 4. one final pass over every candidate
        f["final"] = _observe(p, observe, clock)
    rec["t_verified_ns"] = clock()
    return rec


def complete(rec: dict, t0_ns: int, counters_after: dict) -> dict:
    """After the caller's timed region (untimed): the phase labels, the prewarm count, the outcome, the interval
    from the end of the final pass to the timer start (`t0_ns`, same clock), and the counter snapshot taken
    after the request. Returns `rec`, completed in place."""
    files = rec["files"]
    n_read = sum(1 for f in files if f["prewarm"]["status"] in ("read", "error"))
    if not files:
        phases = dict.fromkeys(("initial", "prewarm", "final"), "not_applicable")
    else:
        phases = {"initial": "done", "prewarm": "done" if n_read else "not_needed", "final": "done"}
    rec.update(phases=phases, prewarmed_files=sum(1 for f in files if f["prewarm"]["status"] == "read"),
               outcome=outcome(files), timer_start_minus_verified_ns=t0_ns - rec["t_verified_ns"],
               counters_after_request=counters_after)
    return rec


# ---- kernel counters

def counters(vmstat_path="/proc/vmstat", meminfo_path="/proc/meminfo") -> dict:
    """One read of the cumulative kernel counters. A read problem is recorded, not raised.
    - status "ok": `drop_pagecache` was read;
    - status "unavailable": the file was read but has no `drop_pagecache` line;
    - status "error": /proc/vmstat could not be read.
    `meminfo_kib` is context only. A failure to read it is recorded there and does not change the status."""
    out = {}
    try:
        with open(vmstat_path) as f:
            text = f.read()
    except OSError as e:
        out.update(status="error", error=_err(e))
    else:
        vals = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in VMSTAT_KEYS and parts[1].isdigit():
                vals[parts[0]] = int(parts[1])
        out.update(status="ok" if RULE_COUNTER in vals else "unavailable", vmstat=vals)
    try:
        with open(meminfo_path) as f:
            mi = {}
            for line in f:
                k, _, rest = line.partition(":")
                if k in MEMINFO_KEYS and rest.split() and rest.split()[0].isdigit():
                    mi[k] = int(rest.split()[0])
        out["meminfo_kib"] = mi
    except OSError as e:
        out["meminfo_kib"] = {"error": _err(e)}
    return out
