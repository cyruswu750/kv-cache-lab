import sys, time, statistics
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab import perf_harness as ph
from pathlib import Path
model, tok = km.load_model()
b = ph.Bench(model, tok, Path("/home/cyrus/projects/kv-cache-lab/.kvcache/m2/probe3"), max_L=1024)
b.prepare(256)
def run(tag, n=10):
    xs = [b.measure("full", 1)["e2e_ns"] / 1e6 for _ in range(n)]
    hs = [b.measure("host", 1)["e2e_ns"] / 1e6 for _ in range(n)]
    print(tag, "full median %.1f" % statistics.median(xs), "host median %.1f" % statistics.median(hs), flush=True)
run("baseline")
print(ph.gpu_state())
run("after 1 nvidia-smi")
time.sleep(5); run("5 s later")
time.sleep(20); run("25 s later")
b.close()
