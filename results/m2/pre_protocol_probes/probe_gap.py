import sys, time, statistics, subprocess, random
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from transformers import DynamicCache
from kvlab.model import prefill
model, tok = km.load_model()
ids = torch.randint(100, 20000, (1, 4352), device="cuda")
_, base = prefill(model, ids[:, :1024], None, 0)
def suffix(S):
    c = DynamicCache(ddp_cache_data=[(l.keys.clone(), l.values.clone()) for l in base.layers], config=model.config)
    torch.cuda.synchronize()
    return lambda: prefill(model, ids[:, 1024:1024 + S], c, 1024)
def clocks():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,pstate,power.draw", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
for _ in range(10): prefill(model, ids[:, :1088], None, 0)
res = {}
trials = [(g, kind) for g in (0.0, 0.02, 0.1, 0.5) for kind in ("full1088", "full257", "suf1", "suf64")] * 8
random.Random(0).shuffle(trials)
for g, kind in trials:
    fn = {"full1088": lambda: prefill(model, ids[:, :1088], None, 0), "full257": lambda: prefill(model, ids[:, :257], None, 0)}.get(kind) or suffix(int(kind[3:]))
    torch.cuda.synchronize(); time.sleep(g)
    a = time.perf_counter(); fn(); torch.cuda.synchronize(); b = time.perf_counter()
    res.setdefault((kind, g), []).append((b - a) * 1e3)
for k in sorted(res): print(k, "median %.1f ms" % statistics.median(res[k]), "min %.1f max %.1f" % (min(res[k]), max(res[k])))
print("clocks now:", clocks())
# back-to-back burst after 1s idle: does latency fall as the clock ramps?
time.sleep(1.0)
xs = []
for i in range(12):
    torch.cuda.synchronize(); a = time.perf_counter(); prefill(model, ids[:, :1088], None, 0); torch.cuda.synchronize(); xs.append(round((time.perf_counter() - a) * 1e3, 1))
print("burst after 1 s idle, full1088:", xs)
