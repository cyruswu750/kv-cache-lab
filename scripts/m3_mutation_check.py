"""M3 mutation check: apply each deliberate restore-backend bug to a scratch copy of the tree and
confirm that at least one test fails (KILLED).

Usage: .venv/bin/python scripts/m3_mutation_check.py <empty scratch dir> [--gpu]
CPU mutants run tests/test_restore_backends_cpu.py. With --gpu, GPU mutants also run the named
tests in tests/gpu/test_m3_restore.py (one GPU process at a time; weights from this project's
.hf_cache, offline).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
TC = "kvlab/tiered_cache.py"
CPU_TEST = "tests/test_restore_backends_cpu.py"
GPU = "tests/gpu/test_m3_restore.py"
MUTANTS = [
    # (name, file, old, new, test command args)
    ("every group placed at offset 0", TC, "        s, C = self.next_chunk * self.C, self.C\n",
     "        s, C = 0, self.C\n", [CPU_TEST]),
    ("chunk order reversed within a placed group", TC, ".copy_(self.X[:g].permute(1, 2, 3, 4, 0, 5, 6))",
     ".copy_(self.X[:g].permute(1, 2, 3, 4, 0, 5, 6).flip(4))", [CPU_TEST]),
    ("last partial group never placed", TC,
     "    def finish(self, loaded: int) -> None:\n        self._place()\n        if self.next_chunk != loaded:",
     "    def finish(self, loaded: int) -> None:\n        if False:", [CPU_TEST]),
    ("cache built from the whole planned destination", TC,
     "        return [(self.Y[li, 0, :, :, :L, :], self.Y[li, 1, :, :, :L, :]) for li in range(self.Y.shape[0])]",
     "        return [(self.Y[li, 0], self.Y[li, 1]) for li in range(self.Y.shape[0])]", [CPU_TEST]),
    ("scratch not bounded by the prefix length", TC, "        g = min(n, group)\n", "        g = group\n", [CPU_TEST]),
    ("placement waits for the end (unbounded group)", TC,
     "        if self.filled == self.X.shape[0]:\n            self._place()",
     "        if self.filled == self.X.shape[0]:\n            self.X = torch.cat([self.X, torch.empty_like(self.X)])",
     [CPU_TEST]),
    ("batched backend ignored (always baseline)", TC, '        if self.restore_backend == "batched":\n',
     '        if False:\n', [CPU_TEST]),
    ("hit counted before the copy", TC,
     "                    copy_into_dest(flat, i)\n                sources.append(\"host\")\n                self.counters.host_hits += 1",
     "                    self.counters.host_hits += 1\n                    copy_into_dest(flat, i)\n                sources.append(\"host\")",
     [CPU_TEST]),
]
GPU_MUTANTS = [
    ("chunk H2D made non_blocking (no sync before lease/staging release)", TC,
     "        self.X[self.filled].copy_(self.tiered._typed(flat))  # contiguous H2D",
     "        self.X[self.filled].copy_(self.tiered._typed(flat), non_blocking=True)  # contiguous H2D",
     ["-m", "gpu and m3", f"{GPU}::test_every_chunk_transfer_is_one_synchronizing_copy"]),
    ("chunk order reversed within a placed group (GPU)", TC, ".copy_(self.X[:g].permute(1, 2, 3, 4, 0, 5, 6))",
     ".copy_(self.X[:g].permute(1, 2, 3, 4, 0, 5, 6).flip(4))",
     ["-m", "gpu and m3", f"{GPU}::test_batched_matches_baseline_on_identical_state"]),
]


def main():
    work = Path(sys.argv[1])
    mutants = MUTANTS + (GPU_MUTANTS if "--gpu" in sys.argv[2:] else [])
    env = {**os.environ, "HF_HOME": str(SRC / ".hf_cache"), "HF_HUB_OFFLINE": "1"}
    results = []
    for name, rel, old, new, args in mutants:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for d in ("kvlab", "tests"):
            shutil.copytree(SRC / d, work / d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(SRC / "pyproject.toml", work)
        p = work / rel
        s = p.read_text()
        assert s.count(old) == 1, (name, s.count(old))
        p.write_text(s.replace(old, new))
        r = subprocess.run([str(SRC / ".venv/bin/python"), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *args],
                           cwd=work, capture_output=True, text=True, timeout=900, env=env)
        last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
        results.append((name, "KILLED" if r.returncode not in (0, 5) else "SURVIVED", last))
        print(f"{results[-1][1]:9s} {name:66s} {last}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{sum(r[1] == 'KILLED' for r in results)}/{len(results)} KILLED")
    sys.exit(0 if all(r[1] == "KILLED" for r in results) else 1)


if __name__ == "__main__":
    main()
