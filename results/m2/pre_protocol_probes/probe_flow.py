import sys, time, statistics, random
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab import perf
from kvlab import perf_harness as ph
from pathlib import Path
model, tok = km.load_model()
b = ph.Bench(model, tok, Path("/home/cyrus/projects/kv-cache-lab/.kvcache/m2/probe4"), max_L=1024)
def probe(tag, L, S):
    xs = [b.measure("full", S)["e2e_ns"] / 1e6 for _ in range(5)]
    print(f"{tag:45s} L={b.L} S={S} full {[round(x,1) for x in xs]}", flush=True)
cases = list(perf.SMOKE)
for L in (256, 1024):
    b.prepare(L)
    for S in (1, 64):
        for path in perf.paths_for(S):
            for _ in range(2):
                b.measure(path, S)
        probe(f"after warmup L={L} S={S} all paths", L, S)
b.prepare(1024); probe("prepare(1024)", 1024, 64)
ph.gpu_state(); probe("gpu_state", 1024, 64)
b.correctness(64); probe("correctness(64)", 1024, 64)
for rep in range(4):
    order = [(64, p) for p in perf.paths_for(64)]
    random.shuffle(order)
    for S, p in order:
        s = b.measure(p, S)
        print(f"   rep {rep} {p:12s} {s['e2e_ns']/1e6:.1f}", flush=True)
probe("after 4 reps", 1024, 64)
b.close()
