import sys, time, statistics, subprocess, bisect
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from kvlab.model import prefill
model, tok = km.load_model()
ids = torch.randint(100, 20000, (1, 1088), device="cuda")
for _ in range(5): prefill(model, ids, None, 0)
mon = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active", "--format=csv,noheader", "-lms", "100"], stdout=subprocess.PIPE, text=True)
t_end = time.time() + 40
samples = []
while time.time() < t_end:
    torch.cuda.synchronize(); w = time.time(); a = time.perf_counter(); prefill(model, ids, None, 0); torch.cuda.synchronize()
    samples.append((w, (time.perf_counter() - a) * 1e3))
    time.sleep(0.03)
mon.terminate(); out = mon.communicate()[0].strip().splitlines()
import datetime
rows = []
for line in out:
    p = [x.strip() for x in line.split(",")]
    try:
        ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        rows.append((ts, int(p[1].split()[0]), int(p[2].split()[0]), p[3], p[4]))
    except Exception: pass
rows.sort(); tss = [r[0] for r in rows]
by_clk = {}
for w, ms in samples:
    i = min(max(bisect.bisect_left(tss, w), 0), len(rows) - 1)
    by_clk.setdefault((rows[i][1], rows[i][2]), []).append(ms)
for k in sorted(by_clk): print("sm,mem MHz", k, "n", len(by_clk[k]), "median %.1f" % statistics.median(by_clk[k]))
print("event reasons seen:", sorted({r[4] for r in rows}))
print("latency quartiles:", [round(x, 1) for x in statistics.quantiles([m for _, m in samples], n=4)])
