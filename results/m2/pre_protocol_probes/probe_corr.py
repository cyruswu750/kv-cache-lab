import sys, time, statistics
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab import perf_harness as ph
from pathlib import Path
model, tok = km.load_model()
b = ph.Bench(model, tok, Path("/home/cyrus/projects/kv-cache-lab/.kvcache/m2/probe2"), max_L=1024)
b.prepare(256)
def run(tag, n=12):
    xs, allocs = [], []
    for _ in range(n):
        m0 = torch.cuda.memory_stats().get("num_device_alloc", -1)
        s = b.measure("full", 1)
        allocs.append(torch.cuda.memory_stats().get("num_device_alloc", -1) - m0)
        xs.append(s["e2e_ns"] / 1e6)
    print(tag, "full median %.1f" % statistics.median(xs), "device allocs/sample", allocs[-3:],
          "reserved MiB", torch.cuda.memory_reserved() >> 20)
run("before correctness")
print(b.correctness(1)["strict_pass"])
run("after correctness")
torch.cuda.empty_cache()
run("after empty_cache")
b.close()
