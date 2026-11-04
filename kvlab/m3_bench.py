"""M3 benchmark harness: baseline vs batched restore backend on identical prepared state
(docs/M3_PROTOCOL.md). Builds on the unchanged M2 Bench (kvlab/perf_harness.py).

- The timed region is the M2 one, unchanged: `Bench.measure()`
      torch.cuda.synchronize(); t0; engine.prefill_request(prompt); int(argmax); t1
  So restore (including the batched staging, placement and the D4 constructor copy), suffix
  prefill and admission are all inside it.
- Both backends use the same stores (A: host, B: disk), the same prepared prefix and the same
  admission reset; only `TieredKVCache.restore_backend` is switched before a sample.
- Untimed per sample: peak allocated/reserved device memory (reset before the sample) and the
  restore's own copy/byte counts (`TieredKVCache.last_restore`).
- Untimed checks per case: M2 strict correctness for each backend, an A/B check of every K/V
  byte, suffix logits and 8 greedy steps between the backends, and a restore-only memory
  measurement.
"""

import gc
import hashlib
import subprocess
import time
from pathlib import Path

import torch

from kvlab import diag_harness as dh
from kvlab import perf
from kvlab.model import greedy_continue
from kvlab.perf_harness import Bench
from kvlab.tiered_cache import RESTORE_BACKENDS

PATHS = ("full", "host", "disk")
MODES = ("plain", "synced")
COMBOS = (("full", "none", "plain"),
          *((p, b, m) for m in MODES for p in ("host", "disk") for b in RESTORE_BACKENDS))


def source_identity(root: Path) -> dict:
    """Git baseline + the exact bytes of every source file an M3 experiment can execute."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout
    files = sorted([*root.glob("kvlab/*.py"), *root.glob("scripts/*.py"), *root.glob("tests/**/*.py"),
                    root / "pyproject.toml"])
    diff = git("diff", "HEAD", "--binary")
    return {"git_head": git("rev-parse", "HEAD").strip(), "git_status_porcelain": git("status", "--porcelain").splitlines(),
            "git_diff_head_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def expected_copy_calls(backend: str, n: int, num_layers: int) -> int:
    return 2 * num_layers * n if backend == "baseline" else n


def sample_problems(s: dict, num_layers: int = 28) -> list[str]:
    """M2 exclusion rules plus: the requested backend actually ran, with the expected copy count."""
    problems = perf.sample_problems({**s, "instrumented": s["mode"] == "synced"})
    lr = s.get("last_restore")
    if s["path"] == "full":
        if lr is not None:
            problems.append(f"full path ran a restore: {lr}")
    else:
        n = s["L"] // perf.C
        if not lr:
            problems.append("no restore record")
        else:
            if lr.get("backend") != s["backend"]:
                problems.append(f"backend {lr.get('backend')!r} != requested {s['backend']!r}")
            if lr.get("chunks_planned") != n or lr.get("chunks_loaded") != n or \
                    lr.get("h2d_bytes") != lr.get("device_destination_bytes"):
                problems.append(f"restore record {lr}")
            if lr.get("h2d_copy_calls") != expected_copy_calls(s["backend"], n, num_layers):
                problems.append(f"h2d_copy_calls {lr.get('h2d_copy_calls')} for backend {s['backend']}")
    return problems


class M3Bench(Bench):
    def __init__(self, *a, staging_chunks: int = 4, **kw):
        super().__init__(*a, **kw)
        for t in (self.tiered_A, self.tiered_B):
            t.restore_staging_chunks = staging_chunks
            t._check_backend()

    def set_backend(self, backend: str) -> None:
        for t in (self.tiered_A, self.tiered_B):
            t.restore_backend = "baseline" if backend == "none" else backend
            t.last_restore = None

    def measure_m3(self, path: str, S: int, backend: str, mode: str) -> dict:
        if path not in PATHS or mode not in MODES or (path == "full") != (backend == "none"):
            raise ValueError(f"bad combination {path}/{backend}/{mode}")
        self.set_backend(backend)
        tiered = self.tiered_B if path == "disk" else self.tiered_A
        torch.cuda.synchronize()
        alloc0, reserved0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        ru0, a0 = dh._rusage(), dh._alloc()
        rec = self.measure(path, S, instrumented=(mode == "synced"))  # the unchanged M2 timed region
        ru1, a1 = dh._rusage(), dh._alloc()
        rec.pop("instrumented")
        rec.update(backend=backend, mode=mode, last_restore=tiered.last_restore,
                   mem={"allocated_before": alloc0, "reserved_before": reserved0,
                        "peak_allocated": torch.cuda.max_memory_allocated(),
                        "peak_reserved": torch.cuda.max_memory_reserved(),
                        "allocated_after": torch.cuda.memory_allocated()},
                   rusage_delta=dh._delta(ru0, ru1), alloc_delta=dh._delta(a0, a1))
        return rec

    # ---- untimed checks
    def correctness_both(self, S: int) -> dict:
        """M2 strict correctness (bytes vs source, host == disk == resident bitwise) per backend."""
        out = {}
        for b in RESTORE_BACKENDS:
            self.set_backend(b)
            out[b] = self.correctness(S)
        self.set_backend("baseline")
        out["strict_pass_all"] = all(out[b]["strict_pass"] for b in RESTORE_BACKENDS)
        return out

    def ab_check(self, S: int, steps: int = 8) -> dict:
        """Same prompt, same state, one request per backend and tier: every K/V byte after the
        suffix prefill, the first-token logits, `steps` greedy steps and the admissions must match."""
        L = self.L
        prompt = self.base[:L + S]
        out = {"L": L, "S": S}
        for path, eng_key in (("host", "A"), ("disk", "B")):
            seen = {}
            for b in RESTORE_BACKENDS:
                self.set_backend(b)
                p = self.engine[eng_key].prefill_request(prompt, allow_restore=True)
                kv = [(layer.keys.view(torch.uint8).clone(), layer.values.view(torch.uint8).clone())
                      for layer in p.cache.layers]
                toks, step_logits = greedy_continue(self.model, p.cache, p.logits, len(prompt), steps)
                seen[b] = {"restore_len": p.restore_len, "sources": p.sources, "stored": p.stored_chunks,
                           "logits": p.logits.float().cpu(), "kv": kv, "tokens": toks, "steps": step_logits}
                stored = p.stored_chunks
                del p
                self._reset_admissions(path, prompt, stored)
            x, y = seen["baseline"], seen["batched"]
            out[path] = {
                "restore_len_sources_admission_equal": (x["restore_len"], x["sources"], x["stored"]) ==
                                                       (y["restore_len"], y["sources"], y["stored"]),
                "kv_bytes_equal_after_prefill": all(torch.equal(a, c) and torch.equal(b_, d)
                                                    for (a, b_), (c, d) in zip(x["kv"], y["kv"])),
                "first_logits_bitwise_equal": torch.equal(x["logits"], y["logits"]),
                "greedy_tokens_equal": x["tokens"] == y["tokens"],
                "step_logits_bitwise_equal": len(x["steps"]) == len(y["steps"]) == steps and
                                             all(torch.equal(a, c) for a, c in zip(x["steps"], y["steps"])),
                "restore_len": x["restore_len"], "sources": x["sources"]}
            del seen
        self.set_backend("baseline")
        out["pass"] = all(all(v for k, v in out[p].items() if isinstance(v, bool)) for p in ("host", "disk"))
        return out

    def restore_memory(self) -> dict:
        """Restore-only device memory per backend and tier (untimed): peak above the starting
        allocation while restoring the prepared prefix, and what stays allocated (the cache)."""
        n = self.L // perf.C
        out = {}
        for path, t in (("host", self.tiered_A), ("disk", self.tiered_B)):
            if path == "disk":
                self.page_cache_before("disk")
            for b in RESTORE_BACKENDS:
                self.set_backend(b)
                gc.collect()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                cache, L, sources, why = t.restore(self.base[:self.L + 1], self.keys[:n], self.device, self.model.config)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                kept = torch.cuda.memory_allocated() - base
                out[f"{path}_{b}"] = {"L": L, "sources_ok": sources == [path] * n, "peak_above_start": peak,
                                      "kept_after_return": kept, "last_restore": t.last_restore,
                                      "reserved_after": torch.cuda.memory_reserved()}
                del cache
        self.set_backend("baseline")
        return out
