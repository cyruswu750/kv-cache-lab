"""M2.1 diagnostic harness (docs/M2_1_DIAGNOSTIC_PLAN.md). Diagnostic only: nothing here
feeds the M2 cost model or the historical M2 scores.

Paths:
  full, host, disk  - as in M2 (kvlab/perf_harness.Bench). "full" = recompute from the
                      prepared cache state: prefix chunks are cached in store A, restore is
                      not attempted, admission of new complete chunks is unchanged.
  full_nocache      - no tiered cache at all: token ids -> device, kvlab.model.prefill with
                      an empty cache, argmax. No chain keys, lookup or admission.
Modes:
  plain   - the unchanged M2 timed region (Bench.measure, uninstrumented)
  synced  - the unchanged M2 instrumented region (device sync at every phase boundary)
  events  - no device sync inside the region; at each phase boundary a pre-created CUDA
            event is recorded and the wall clock and thread CPU time are read
Every mode keeps the M2 boundary: torch.cuda.synchronize(); t0 ... int(argmax); t1.
"""

import gc
import hashlib
import importlib.metadata
import os
import resource
import threading
import time
import warnings
from pathlib import Path

import torch
from transformers import DynamicCache

from kvlab import perf
from kvlab.model import prefill
from kvlab.perf_harness import Bench

DIAG_PATHS = ("full", "host", "disk", "full_nocache")
PROBES = ("probe_cpu_sha", "probe_cpu_py", "probe_launch", "probe_copy",       # Amendment 1
          "probe_syscall_getppid", "probe_syscall_read")                     # Amendment 2
MODES = ("plain", "events", "synced")
ENGINE_PHASES = ("ingest", "restore", "prefill", "store", "select_launched")
NOCACHE_PHASES = ("ingest", "prefill", "select_launched")
ALLOC_KEYS = ("num_device_alloc", "num_device_free", "num_alloc_retries")
RUSAGE_KEYS = ("ru_utime", "ru_stime", "ru_nvcsw", "ru_nivcsw", "ru_minflt", "ru_majflt")


def expected(path: str, L: int, S: int) -> dict:
    if path == "full_nocache":
        perf.check_case(L, S)
        return {"restore_len": 0, "sources": [], "stored_chunks": []}
    return perf.expected(path, L, S)


def sample_problems(s: dict) -> list[str]:
    """Exclusion reasons for a diagnostic sample (M2 rules plus the events/nocache fields)."""
    if s["path"] in PROBES:
        ok = isinstance(s.get("e2e_ns"), int) and s["e2e_ns"] > 0 and s.get("probe_check") is True
        return [] if ok else [f"probe e2e_ns={s.get('e2e_ns')!r} check={s.get('probe_check')!r}"]
    if s["path"] != "full_nocache":
        base = {k: v for k, v in s.items() if k not in ("instrumented", "phases_ns")}
        problems = perf.sample_problems({**base, "instrumented": s["mode"] == "synced",
                                         "phases_ns": s.get("phases_ns")})
    else:
        problems = [f"{k}={s.get(k)!r}, expected {v!r}" for k, v in expected("full_nocache", s["L"], s["S"]).items()
                    if s.get(k) != v]
        if not isinstance(s.get("e2e_ns"), int) or s["e2e_ns"] <= 0:
            problems.append(f"e2e_ns={s.get('e2e_ns')!r}")
        if s["mode"] == "synced" and set(s.get("phases_ns") or {}) != {"ingest", "prefill", "select"}:
            problems.append(f"phases_ns={s.get('phases_ns')!r}")
    if s["mode"] == "events":
        want = NOCACHE_PHASES if s["path"] == "full_nocache" else ENGINE_PHASES
        ev = s.get("events") or {}
        if [m["phase"] for m in ev.get("marks", [])] != list(want):
            problems.append(f"event marks {[m.get('phase') for m in ev.get('marks', [])]} != {list(want)}")
        elif any(m["gpu_ms"] is None or m["gpu_ms"] < 0 for m in ev["marks"]):
            problems.append("negative or missing event time")
    return problems


def _rusage() -> dict:
    """Thread rusage plus /proc schedstat (ns on CPU, ns waiting on a runqueue, timeslices)."""
    r = resource.getrusage(resource.RUSAGE_THREAD)
    run_ns, wait_ns, slices = (int(x) for x in
                               Path(f"/proc/self/task/{threading.get_native_id()}/schedstat").read_text().split())
    return {**{k: getattr(r, k) for k in RUSAGE_KEYS}, "sched_run_ns": run_ns, "sched_runqueue_wait_ns": wait_ns,
            "sched_timeslices": slices}


def _delta(a: dict, b: dict) -> dict:
    return {k: b[k] - a[k] for k in a}


def _alloc() -> dict:
    st = torch.cuda.memory_stats()
    return {k: int(st.get(k, 0)) for k in ALLOC_KEYS}


def package_versions() -> dict:
    out = {}
    for d in importlib.metadata.distributions():
        name = d.metadata["Name"]
        if name and (name.lower().startswith(("nvidia", "torch", "transformers", "numpy", "safetensors",
                                              "tokenizers", "huggingface", "triton", "accelerate"))):
            out[name] = d.version
    return dict(sorted(out.items()))


def weight_blobs(project_root: Path, model_id: str, revision: str) -> dict:
    """HF cache snapshot files -> blob names (LFS blobs are named by the content sha256)."""
    snap = project_root / ".hf_cache" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not snap.is_dir():
        return {"error": f"no snapshot dir {snap}"}
    return {p.name: {"blob": os.path.basename(os.readlink(p)) if p.is_symlink() else None, "bytes": p.stat().st_size}
            for p in sorted(snap.iterdir())}


def process_record() -> dict:
    import torch.backends.cuda as tbc
    st = Path("/proc/self/status").read_text().splitlines()
    return {
        "pid": os.getpid(), "nice": os.nice(0), "affinity_n": len(os.sched_getaffinity(0)), "nproc": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_LAUNCH_BLOCKING",
                                                 "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING", "HF_HUB_OFFLINE")},
        "rlimit_memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "meminfo_head": Path("/proc/meminfo").read_text().splitlines()[:6],
        "status_threads": next((ln for ln in st if ln.startswith("Threads")), None),
        "cpufreq_visible": Path("/sys/devices/system/cpu/cpu0/cpufreq").exists(),
        "sdpa_backends": {"flash": tbc.flash_sdp_enabled(), "mem_efficient": tbc.mem_efficient_sdp_enabled(),
                          "math": tbc.math_sdp_enabled(), "cudnn": tbc.cudnn_sdp_enabled()},
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


class DiagBench(Bench):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Pre-created and pre-recorded events: no event creation inside a timed region.
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        for e in self.events:
            e.record()
        torch.cuda.synchronize()

    # ---- Amendment 1 probes: fixed CPU-only work vs fixed CUDA-call work
    def _probe_setup(self):
        if getattr(self, "_probe_buf", None) is None:
            self._probe_buf = bytes(range(256)) * (8 * 1024 * 1024 // 256)  # 8 MiB
            self._probe_sha = hashlib.sha256(self._probe_buf).hexdigest()
            self._probe_dev = torch.zeros(1, device=self.device)
            self._probe_pinned = torch.ones(56 * 131072, dtype=torch.uint8).pin_memory()
            self._probe_dst = torch.empty(56 * 131072, dtype=torch.uint8, device=self.device)
            self._probe_zero_fd = os.open("/dev/zero", os.O_RDONLY)
            torch.cuda.synchronize()

    def measure_probe(self, name: str) -> dict:
        self._probe_setup()
        v0 = float(self._probe_dev.item())
        self._probe_dst.zero_()
        torch.cuda.synchronize()
        ru0 = _rusage()
        gc.disable()
        try:
            torch.cuda.synchronize()
            t_wall = time.time()
            c0 = time.thread_time_ns()
            t0 = time.perf_counter_ns()
            if name == "probe_cpu_sha":
                check = hashlib.sha256(self._probe_buf).hexdigest()
            elif name == "probe_cpu_py":
                acc = 0
                for i in range(300_000):
                    acc += i * i % 7
                check = acc
            elif name == "probe_syscall_getppid":
                for _ in range(20_000):
                    os.getppid()
                check = None
            elif name == "probe_syscall_read":
                n = 0
                for _ in range(5_000):
                    n += len(os.read(self._probe_zero_fd, 4096))
                check = n
            elif name == "probe_launch":
                for _ in range(500):
                    self._probe_dev.add_(1.0)
                torch.cuda.synchronize()
                check = None
            elif name == "probe_copy":
                for i in range(56):
                    self._probe_dst[i * 131072:(i + 1) * 131072].copy_(self._probe_pinned[i * 131072:(i + 1) * 131072])
                check = None
            else:
                raise ValueError(name)
            t1 = time.perf_counter_ns()
            c1 = time.thread_time_ns()
        finally:
            gc.enable()
        ru1 = _rusage()
        ok = {"probe_cpu_sha": lambda: check == self._probe_sha, "probe_cpu_py": lambda: check == 599_998,
              "probe_syscall_getppid": lambda: True, "probe_syscall_read": lambda: check == 5_000 * 4096,
              "probe_launch": lambda: float(self._probe_dev.item()) - v0 == 500.0,
              "probe_copy": lambda: bool(torch.equal(self._probe_dst, self._probe_pinned.to(self.device)))}[name]()
        return {"path": name, "mode": "plain", "L": self.L, "S": None, "e2e_ns": t1 - t0, "t_wall": t_wall,
                "t0_mono_ns": t0, "thread_cpu_ns": c1 - c0, "probe_check": bool(ok),
                "rusage_delta": _delta(ru0, ru1), "alloc_delta": {k: 0 for k in ALLOC_KEYS}}

    # ---- the no-cache recompute baseline
    def _nocache_region(self, prompt, mark):
        ids = torch.tensor([prompt], device=self.device)
        mark("ingest")
        logits, cache = prefill(self.model, ids, None, 0)
        mark("prefill")
        idx = torch.argmax(logits)
        mark("select_launched")
        token = int(idx)
        return token, cache

    def measure_diag(self, path: str, S: int, mode: str) -> dict:
        if path not in DIAG_PATHS or mode not in MODES:
            raise ValueError(f"unknown path/mode {path}/{mode}")
        if path != "full_nocache" and mode in ("plain", "synced"):
            ru0, a0 = _rusage(), _alloc()
            rec = self.measure(path, S, instrumented=(mode == "synced"))  # the unchanged M2 code
            ru1, a1 = _rusage(), _alloc()
            rec.pop("instrumented")
            return {**rec, "mode": mode, "rusage_delta": _delta(ru0, ru1), "rusage_scope": "around_call",
                    "alloc_delta": _delta(a0, a1)}
        return self._measure_own(path, S, mode)

    def _measure_own(self, path: str, S: int, mode: str) -> dict:
        L = self.L
        prompt = self.base[:L + S]
        extra = self.page_cache_before(path)
        evs, stamps = self.events, []

        if mode == "events":
            def mark(phase):
                evs[len(stamps) + 1].record()
                stamps.append((phase, time.perf_counter_ns(), time.thread_time_ns()))
        elif mode == "synced":
            def mark(phase):
                torch.cuda.synchronize()
                stamps.append((phase, time.perf_counter_ns(), time.thread_time_ns()))
        else:
            def mark(phase):
                pass
        a0 = _alloc()
        ru0 = _rusage()
        gc.disable()
        try:
            torch.cuda.synchronize()
            t_wall = time.time()
            c0 = time.thread_time_ns()
            t0 = time.perf_counter_ns()
            if mode == "events":
                evs[0].record()
            if path == "full_nocache":
                token, cache = self._nocache_region(prompt, mark)
                info = {"restore_len": 0, "sources": [], "stored_chunks": [], "miss_reason": "no_cache_system"}
            else:
                eng = self.engine["B" if path.startswith("disk") else "A"]
                p = eng.prefill_request(prompt, allow_restore=(path != "full"), mark=mark)
                idx = torch.argmax(p.logits)
                mark("select_launched")
                token = int(idx)
                info = {"restore_len": p.restore_len, "sources": p.sources, "stored_chunks": p.stored_chunks,
                        "miss_reason": p.miss_reason}
            t1 = time.perf_counter_ns()
            c1 = time.thread_time_ns()
        finally:
            gc.enable()
        ru1 = _rusage()
        rec = {"path": path, "L": L, "S": S, "N": L + S, "mode": mode, "e2e_ns": t1 - t0, "t_wall": t_wall,
               "t0_mono_ns": t0, "thread_cpu_ns": c1 - c0, **info, "first_token": token, **extra,
               "rusage_delta": _delta(ru0, ru1), "rusage_scope": "timed_region"}
        if mode == "events":
            torch.cuda.synchronize()
            marks, prev_w, prev_c = [], t0, c0
            for i, (phase, tw, tc) in enumerate(stamps):
                marks.append({"phase": phase, "wall_ns": tw - prev_w, "thread_cpu_ns": tc - prev_c,
                              "wall_from_t0_ns": tw - t0, "gpu_ms": evs[0].elapsed_time(evs[i + 1])})
                prev_w, prev_c = tw, tc
            rec["events"] = {"marks": marks, "final_wait_wall_ns": t1 - prev_w, "final_wait_thread_cpu_ns": c1 - prev_c}
        elif mode == "synced":
            prev, ph = t0, {}
            for phase, tw, _ in stamps:
                ph["select" if phase == "select_launched" else phase] = tw - prev
                prev = tw
            ph["select"] = ph.get("select", 0) + (t1 - prev)
            rec["phases_ns"] = ph
        if path == "full_nocache":
            del cache
        else:
            del p
            self._reset_admissions(path, prompt, rec["stored_chunks"])
        rec["alloc_delta"] = _delta(a0, _alloc())
        return rec

    # ---- untimed checks
    def nocache_matches_full(self, S: int) -> dict:
        """full_nocache logits bitwise equal to the engine's full-recompute logits (same ids)."""
        prompt = self.base[:self.L + S]
        p = self.engine["A"].prefill_request(prompt, allow_restore=False)
        full = p.logits.float().cpu()
        stored = p.stored_chunks
        del p
        self._reset_admissions("full", prompt, stored)
        with torch.inference_mode():
            lg, cache = prefill(self.model, torch.tensor([prompt], device=self.device), None, 0)
        nc = lg.float().cpu()
        del cache
        return {"L": self.L, "S": S, "bitwise_equal": bool(torch.equal(full, nc)),
                "first_token_equal": int(full.argmax()) == int(nc.argmax())}

    def spike_probe_audit(self, n: int = 20) -> dict:
        """Re-run the M2 spike probe's cached-forward function with explicit checks that every
        repetition starts at 256 cached tokens and leaves the source K/V untouched."""
        dev, C = self.device, perf.C
        _, cache256 = prefill(self.model, torch.tensor([self.base[:C]], device=dev), None, 0)
        kv = [(layer.keys.clone(), layer.values.clone()) for layer in cache256.layers]  # as in m2_spike_probe.py
        del cache256
        suffix_ids = torch.tensor([self.base[C:C + 1]], device=dev)

        def digest():
            h = hashlib.sha256()
            for k, v in kv:
                h.update(k.cpu().view(torch.uint8).numpy().tobytes())
                h.update(v.cpu().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        ptrs0, shapes0, d0 = [(k.data_ptr(), v.data_ptr()) for k, v in kv], [tuple(k.shape) for k, _ in kv], digest()
        lens = []
        for _ in range(n):
            c = DynamicCache(ddp_cache_data=kv, config=self.model.config)
            before = c.get_seq_length()
            prefill(self.model, suffix_ids, c, C)
            lens.append((before, c.get_seq_length()))
            del c
        ok_ptr = [(k.data_ptr(), v.data_ptr()) for k, v in kv] == ptrs0
        ok_shape = [tuple(k.shape) for k, _ in kv] == shapes0
        d1 = digest()
        return {"n": n, "cache_len_before_after": sorted(set(lens)), "all_start_at_256": all(b == C for b, _ in lens),
                "all_end_at_257": all(a == C + 1 for _, a in lens), "source_kv_pointers_unchanged": ok_ptr,
                "source_kv_shapes_unchanged": ok_shape, "source_kv_bytes_unchanged": d0 == d1,
                "pass": all(b == C and a == C + 1 for b, a in lens) and ok_ptr and ok_shape and d0 == d1}

    def sync_debug(self) -> dict:
        """Which host synchronizations happen inside a cached forward and a full forward
        (torch sync debug mode, untimed). Messages are de-duplicated."""
        dev, C = self.device, perf.C
        out = {}
        _, cache256 = prefill(self.model, torch.tensor([self.base[:C]], device=dev), None, 0)
        kv = [(layer.keys.clone(), layer.values.clone()) for layer in cache256.layers]
        del cache256
        for name, fn in (("cached_forward_S1_L256", lambda: prefill(self.model, torch.tensor([self.base[C:C + 1]], device=dev),
                                                                     DynamicCache(ddp_cache_data=kv, config=self.model.config), C)),
                         ("full_forward_N257", lambda: prefill(self.model, torch.tensor([self.base[:C + 1]], device=dev), None, 0))):
            torch.cuda.synchronize()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    fn()
                finally:
                    torch.cuda.set_sync_debug_mode(0)
            msgs = [str(x.message).splitlines()[0][:200] for x in w]
            out[name] = {"n_sync_warnings": len(msgs), "distinct": sorted(set(msgs))}
        return out
