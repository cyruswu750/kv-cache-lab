"""M2 GPU benchmark harness (docs/M2_PROTOCOL.md).

Timed region of one sample (`Bench.measure`):
    torch.cuda.synchronize(); t0
    engine.prefill_request(prompt)   # ids to device, chain keys, restore (tier), prefill, store
    int(torch.argmax(logits))        # first token on the host (implies a device sync)
    t1
Everything else - state preparation, page-cache handling, resets, checks - is outside it.
Tier states (prefix [0, L) cached; exclusive tiers as in M1):
    store A: prefix chunks in pinned host slots (paths "full" and "host")
    store B: prefix chunks as files on disk, host holds no prefix chunk (paths "disk", "disk_evicted")
Each store keeps one free host slot so that admitting a new suffix chunk (S = 256) never spills;
admitted chunks are discarded after each sample.
"""

import atexit
import ctypes
import gc
import hashlib
import mmap
import os
import signal
import subprocess
import time
import zlib
from pathlib import Path

import torch
from transformers import DynamicCache

import kvlab.model as km
from kvlab import chunk_format as cf
from kvlab import perf
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.fixtures import BOUNDARY_TEXT, CALIBRATION_TEXT, exact_length_ids
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.kv_snapshot import bytes_equal
from kvlab.model import prefill
from kvlab.tiered_cache import TieredKVCache

C = perf.C
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_PAGE = os.sysconf("SC_PAGE_SIZE")


def resident_fraction(path) -> float:
    """Fraction of the file's pages in the Linux page cache (mincore on an untouched mapping)."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    try:
        n = (size + _PAGE - 1) // _PAGE
        vec = (ctypes.c_ubyte * n)()
        buf = ctypes.c_char.from_buffer(mm)
        try:
            if _libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
        finally:
            del buf
        return sum(b & 1 for b in vec) / n
    finally:
        mm.close()


def evict_from_page_cache(path) -> None:
    """Drop the clean cached pages of one of our own (fsynced) files; no global cache drop."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def warm_page_cache(path) -> None:
    with open(path, "rb", buffering=0) as f:
        while f.read(1 << 22):
            pass


GPU_MONITOR_QUERY = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,utilization.gpu," \
                    "clocks_event_reasons.active"


def start_gpu_monitor(csv_path: Path, period_ms: int = 200):
    """Read-only nvidia-smi polling in a separate process (no settings are changed)."""
    f = open(csv_path, "w")
    proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_MONITOR_QUERY}", "--format=csv,noheader",
                             "-lms", str(period_ms)], stdout=f, stderr=subprocess.DEVNULL,
                            preexec_fn=lambda: _libc.prctl(1, signal.SIGTERM))  # PR_SET_PDEATHSIG
    handle = (proc, f)
    atexit.register(stop_gpu_monitor, handle)
    return handle


def stop_gpu_monitor(handle) -> None:
    proc, f = handle
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    f.close()


def gpu_state() -> dict:
    q = "temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.max.sm,pstate,utilization.gpu,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": repr(e)}
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **dict(zip(q.split(","), [x.strip() for x in out.split(",")]))}


def source_manifest(root: Path) -> dict:
    files = sorted([*root.glob("kvlab/*.py"), *root.glob("scripts/m2_*.py")])
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def environment(root: Path, disk_root: Path) -> dict:
    import platform
    import transformers
    fs = subprocess.run(["df", "-T", str(disk_root)], capture_output=True, text=True).stdout.strip().splitlines()
    mounts = [line for line in Path("/proc/mounts").read_text().splitlines() if line.split()[1] == "/"]
    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
            "gpu_static": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm,"
                                          "clocks.max.mem,memory.total", "--format=csv"], capture_output=True,
                                         text=True).stdout.strip(),
            "cpu": next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                         if ln.startswith("model name")), None),
            "kernel": platform.release(), "disk_root": str(disk_root), "filesystem_df": fs, "root_mount": mounts,
            "page_size": _PAGE, "source_manifest": source_manifest(root)}


class Bench:
    def __init__(self, model, tok, disk_root: Path, max_L: int = 4096, max_S: int = 256):
        self.model = model
        self.device = next(model.parameters()).device
        self.ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, C)
        self.digest = self.ns.digest()
        cb, fsize = cf.payload_len(self.ns), cf.file_size(self.ns)
        self.chunk_bytes, self.file_bytes = cb, fsize
        self.n_max = max_L // C
        n_adm = max_S // C  # new complete chunks one request can admit (S <= 256 -> at most 1)
        self.base = exact_length_ids(tok, BOUNDARY_TEXT + "\n\n" + CALIBRATION_TEXT, max_L + max_S)
        self.keys = chain_keys(self.digest, self.base[:max_L], C)
        self.idents = [ChunkIdentity.for_chunk(self.digest, self.base, C, i, self.keys) for i in range(self.n_max)]
        # Source of chunk bytes: one untimed full prefill of the longest prefix, stored by the M1 code path.
        self.src_host = HostStore((self.n_max + 1) * cb, cb, self.n_max, pin=False)
        src = TieredKVCache(self.ns, self.src_host, None)
        with torch.inference_mode():
            _, cache = prefill(model, torch.tensor([self.base[:max_L]], device=self.device), None, 0)
        assert src.store(cache, self.base[:max_L], self.keys) == list(range(self.n_max))
        del cache
        self.host_A = HostStore((self.n_max + n_adm + 1) * cb, cb, self.n_max + n_adm, pin=True)
        self.tiered_A = TieredKVCache(self.ns, self.host_A, None)
        self.host_B = HostStore((n_adm + 1) * cb, cb, n_adm, pin=True)
        self.disk_B = DiskStore(disk_root, (self.n_max + n_adm) * fsize, self.n_max + n_adm)
        self.tiered_B = TieredKVCache(self.ns, self.host_B, self.disk_B)
        self.engine = {"A": AlwaysRestoreEngine(model, self.tiered_A), "B": AlwaysRestoreEngine(model, self.tiered_B)}
        self.L = None

    # ---- state
    def _no_spill(self, *a):
        raise RuntimeError("benchmark state must never spill")

    def prepare(self, L: int) -> None:
        """Make store A hold exactly prefix chunks [0, L) in host, and store B exactly the same
        chunks on disk (fsynced files, so their pages can later be evicted), host B empty."""
        n = L // C
        want = self.keys[:n]
        for k in list(self.host_A.entries):
            if k not in want:
                self.host_A.discard(k)
        for k in list(self.host_B.entries):
            self.host_B.discard(k)
        for i, k in enumerate(want):
            if k not in self.host_A:
                slot, why = self.host_A.allocate(self._no_spill)
                assert why == "ok", why
                with self.src_host.lease(k) as (flat, _):
                    self.host_A.slot_view(slot).copy_(flat)
                self.host_A.insert(k, self.idents[i], slot)
        if set(self.disk_B.entries) != set(want):
            self.disk_B.clear()
            for i, k in enumerate(want):
                with self.src_host.lease(k) as (flat, _):
                    ok, why = self.disk_B.put(k, cf.encode_header(self.ns, self.idents[i]), flat.numpy())
                assert ok, why
                fd = os.open(self.chunk_path(i), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        self.L = L
        self.check_state()

    def check_state(self) -> None:
        want = set(self.keys[:self.L // C])
        assert set(self.host_A.entries) == want, "host A state drifted"
        assert not self.host_B.entries, "host B must hold no chunk"
        assert set(self.disk_B.entries) == want and not self.disk_B.undeletable, "disk B state drifted"

    def chunk_path(self, i: int) -> Path:
        return self.disk_B.root / f"{self.keys[i].hex()}.kvc"

    def disk_files(self) -> list[Path]:
        return [self.chunk_path(i) for i in range(self.L // C)]

    def _reset_admissions(self, path: str, prompt, stored) -> None:
        if stored:
            keys = chain_keys(self.digest, prompt, C)
            host = self.host_B if path.startswith("disk") else self.host_A
            for i in stored:
                host.discard(keys[i])
        self.check_state()

    # ---- one sample
    def page_cache_before(self, path: str) -> dict:
        """Untimed page-cache handling before a disk-path sample (evict or verify/warm)."""
        extra = {}
        if path == "disk_evicted":
            for f in self.disk_files():
                evict_from_page_cache(f)
            extra["resident_fraction_before"] = max(resident_fraction(f) for f in self.disk_files())
        elif path == "disk":
            frac = min(resident_fraction(f) for f in self.disk_files())
            if frac < 1.0:
                for f in self.disk_files():
                    warm_page_cache(f)
                extra["prewarmed_from_fraction"] = frac
                frac = min(resident_fraction(f) for f in self.disk_files())
            extra["resident_fraction_before"] = frac
        return extra

    def measure(self, path: str, S: int, instrumented: bool = False) -> dict:
        L = self.L
        prompt = self.base[:L + S]
        eng = self.engine["B" if path.startswith("disk") else "A"]
        extra = self.page_cache_before(path)
        stamps = []

        def mark(phase):
            torch.cuda.synchronize()
            stamps.append((phase, time.perf_counter_ns()))
        gc.disable()  # no collector pause inside the timed region (callers collect between repetitions)
        try:
            torch.cuda.synchronize()
            t_wall = time.time()
            t0 = time.perf_counter_ns()
            p = eng.prefill_request(prompt, allow_restore=(path != "full"), mark=mark if instrumented else None)
            token = int(torch.argmax(p.logits))
            t1 = time.perf_counter_ns()
        finally:
            gc.enable()
        rec = {"path": path, "L": L, "S": S, "N": L + S, "instrumented": instrumented, "e2e_ns": t1 - t0,
               "t_wall": t_wall,
               "restore_len": p.restore_len, "sources": p.sources, "stored_chunks": p.stored_chunks,
               "miss_reason": p.miss_reason, "first_token": token, **extra}
        if instrumented:
            prev, phases = t0, {}
            for phase, t in stamps:
                phases[phase] = t - prev
                prev = t
            phases["select"] = t1 - prev
            rec["phases_ns"] = phases
        del p
        self._reset_admissions(path, prompt, rec["stored_chunks"])
        return rec

    # ---- correctness (separate from timing; D10/D12 distinctions)
    def correctness(self, S: int) -> dict:
        L, n = self.L, self.L // C
        prompt = self.base[:L + S]
        bytes_ok = {}

        def checker(tag):
            def on_restored(cache, Lr):
                ok = Lr == L
                for i in range(n):
                    with self.src_host.lease(self.keys[i]) as (flat, _):
                        src = self.tiered_A._typed(flat)
                        for li, layer in enumerate(cache.layers):
                            ok &= bytes_equal(layer.keys[:, :, i * C:(i + 1) * C, :].cpu(), src[li, 0])
                            ok &= bytes_equal(layer.values[:, :, i * C:(i + 1) * C, :].cpu(), src[li, 1])
                bytes_ok[tag] = bool(ok)
            return on_restored
        out = {"L": L, "S": S}
        logits = {}
        for tag, eng_key, allow in (("host", "A", True), ("disk", "B", True), ("full", "A", False)):
            p = self.engine[eng_key].prefill_request(prompt, allow_restore=allow,
                                                     on_restored=checker(tag) if allow else None)
            out[f"{tag}_restore_len"], out[f"{tag}_sources"] = p.restore_len, p.sources
            logits[tag] = p.logits.float().cpu()
            stored = p.stored_chunks
            del p
            self._reset_admissions("disk" if tag == "disk" else tag, prompt, stored)
        # Resident split reference: the same source bytes as GPU tensors, never through a tier.
        data = []
        for li in range(self.ns.num_layers):
            ks, vs = [], []
            for i in range(n):
                with self.src_host.lease(self.keys[i]) as (flat, _):
                    src = self.tiered_A._typed(flat)
                    ks.append(src[li, 0].to(self.device))
                    vs.append(src[li, 1].to(self.device))
            data.append((torch.cat(ks, dim=-2), torch.cat(vs, dim=-2)))
        cache = DynamicCache(ddp_cache_data=data, config=self.model.config)
        with torch.inference_mode():
            lr, cache = prefill(self.model, torch.tensor([prompt[L:]], device=self.device), cache, L)
        logits["resident"] = lr.float().cpu()
        del cache, data
        out["restored_kv_bytes_equal"] = bytes_ok
        out["strict_host_eq_disk_bitwise"] = torch.equal(logits["host"], logits["disk"])
        out["strict_host_eq_resident_bitwise"] = torch.equal(logits["host"], logits["resident"])
        out["strict_pass"] = (out["strict_host_eq_disk_bitwise"] and out["strict_host_eq_resident_bitwise"]
                              and bytes_ok.get("host") is True and bytes_ok.get("disk") is True
                              and out["host_sources"] == ["host"] * n and out["disk_sources"] == ["disk"] * n)
        out["report_only_full_vs_restored"] = {
            "first_token_equal": int(logits["full"].argmax()) == int(logits["host"].argmax()),
            "max_abs_logit_diff": float((logits["full"] - logits["host"]).abs().max())}
        return out

    # ---- component micro-benchmarks (diagnostic; one chunk; warm page cache)
    def micro(self, reps: int = 30, warmup: int = 5) -> dict:
        assert self.L and self.L >= C
        key, ident = self.keys[0], self.idents[0]
        path0 = self.chunk_path(0)
        ns = self.ns
        dest = [(torch.empty((1, ns.num_kv_heads, C, ns.head_dim), dtype=torch.bfloat16, device=self.device),
                 torch.empty((1, ns.num_kv_heads, C, ns.head_dim), dtype=torch.bfloat16, device=self.device))
                for _ in range(ns.num_layers)]

        def h2d():
            with self.host_A.lease(key) as (flat, _):
                src = self.tiered_A._typed(flat)
                for li, (k, v) in enumerate(dest):
                    k.copy_(src[li, 0])
                    v.copy_(src[li, 1])
            torch.cuda.synchronize()

        def disk_read_validate():
            with self.host_B.staging() as st:
                ok, why, _ = self.disk_B.read_into(key, ns, ident, st.numpy())
            assert ok, why

        def raw_read():
            with self.host_B.staging() as st, open(path0, "rb", buffering=0) as f:
                f.seek(self.file_bytes - self.chunk_bytes - 4)
                f.readinto(memoryview(st.numpy()))

        def crc32_only():
            with self.host_B.staging() as st:
                zlib.crc32(memoryview(st.numpy()))

        def assemble():  # the D2D copy DynamicCache makes of restored tensors (decisions D4)
            DynamicCache(ddp_cache_data=dest, config=self.model.config)
            torch.cuda.synchronize()
        out = {"note": "one chunk = 7,340,032 payload bytes; warm page cache; wall clock incl. sync"}
        for name, fn in (("host_chunk_h2d_56_copies", h2d), ("disk_read_into_validated", disk_read_validate),
                         ("raw_payload_readinto", raw_read), ("zlib_crc32_payload", crc32_only),
                         ("dynamic_cache_assembly_1_chunk", assemble)):
            ts = []
            for r in range(warmup + reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                fn()
                t1 = time.perf_counter_ns()
                if r >= warmup:
                    ts.append(t1 - t0)
            out[name] = {"samples_ns": ts, **perf.summarize([perf.ns_to_ms(t) for t in ts])}
            if name != "dynamic_cache_assembly_1_chunk":
                out[name]["effective_GBps_at_median"] = self.chunk_bytes / (out[name]["median_ms"] * 1e6)
        return out

    def budgets(self) -> dict:
        return {"store_A": self.tiered_A.budgets(), "store_B": self.tiered_B.budgets(),
                "source_pageable_slab_bytes": self.src_host.slab_bytes}

    def close(self) -> None:
        self.disk_B.clear()
        self.host_A.close()
        self.host_B.close()
        self.src_host.close()
