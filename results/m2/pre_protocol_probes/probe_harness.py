import sys, time, statistics, random, gc, subprocess
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab import perf_harness as ph
from pathlib import Path
model, tok = km.load_model()
b = ph.Bench(model, tok, Path("/home/cyrus/projects/kv-cache-lab/.kvcache/m2/probe"), max_L=1024)
A = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
def busy(ms):
    torch.cuda.synchronize(); t = time.perf_counter()
    while (time.perf_counter() - t) * 1e3 < ms:
        for _ in range(4): A @ A
        torch.cuda.synchronize()
orig_collect = gc.collect
res = {}
for L in (256, 1024):
    b.prepare(L)
    trials = [(mode, path) for mode in ("as_is", "precond50", "precond50_nogc") for path in ("full", "host", "disk")] * 12
    random.Random(1).shuffle(trials)
    for _ in range(6):
        for path in ("full", "host", "disk"): b.measure(path, 1)
    for mode, path in trials:
        gc.collect = (lambda *a: 0) if mode.endswith("nogc") else orig_collect
        if mode.startswith("precond"):
            busy(50)
        s = b.measure(path, 1)
        res.setdefault((L, path, mode), []).append(s["e2e_ns"] / 1e6)
gc.collect = orig_collect
for k in sorted(res): print(k, "median %.1f  min %.1f  max %.1f" % (statistics.median(res[k]), min(res[k]), max(res[k])))
t = time.perf_counter(); gc.collect(); print("gc.collect ms", round((time.perf_counter() - t) * 1e3, 1))
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,pstate,power.draw,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout)
b.close()
