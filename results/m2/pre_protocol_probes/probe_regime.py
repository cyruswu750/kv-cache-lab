import sys, time, statistics, subprocess
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab import perf_harness as ph
from pathlib import Path
model, tok = km.load_model()
b = ph.Bench(model, tok, Path("/home/cyrus/projects/kv-cache-lab/.kvcache/m2/probe6"), max_L=1024)
b.prepare(1024)
KEYS = ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_sync_all_streams", "segment.all.current", "active_bytes.all.current")
def stats(): s = torch.cuda.memory_stats(); return {k: s.get(k) for k in KEYS}
def full_block(tag, n=6):
    xs = []
    s0 = stats()
    for _ in range(n): xs.append(round(b.measure("full", 64)["e2e_ns"] / 1e6, 1))
    s1 = stats()
    clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"{tag:28s} full {xs}  dalloc {s1['num_device_alloc']-s0['num_device_alloc']} segs {s1['segment.all.current']} active {s1['active_bytes.all.current']>>20}MiB  clk {clk}", flush=True)
for _ in range(3): b.measure("full", 64)
full_block("start")
for k in range(6):
    for _ in range(3): b.measure("host", 64)
    full_block(f"after 3 host (x{k+1})")
for _ in range(3): b.measure("disk", 64)
full_block("after 3 disk")
x = torch.zeros(1, device="cuda").cpu()  # one tiny D2H
full_block("after tiny D2H")
torch.cuda.empty_cache(); full_block("after empty_cache")
b.close()
