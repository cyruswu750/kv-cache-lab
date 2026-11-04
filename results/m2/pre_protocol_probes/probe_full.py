import sys, time, statistics
sys.path.insert(0, "/home/cyrus/projects/kv-cache-lab")
import kvlab.model as km
import torch
from transformers import DynamicCache
from kvlab.model import prefill
model, tok = km.load_model()
ids = torch.randint(100, 20000, (1, 4352), device="cuda")
def t(fn, n=12):
    xs = []
    for i in range(n + 3):
        torch.cuda.synchronize(); a = time.perf_counter(); fn(); torch.cuda.synchronize(); b = time.perf_counter()
        if i >= 3: xs.append((b - a) * 1e3)
    return round(statistics.median(xs), 2)
for N in (1, 2, 16, 64, 257, 1088, 4352):
    print("full prefill N=%d" % N, t(lambda: prefill(model, ids[:, :N], None, 0)))
_, base = prefill(model, ids[:, :256], None, 0)
def with_cache(S):
    c = DynamicCache(ddp_cache_data=[(l.keys.clone(), l.values.clone()) for l in base.layers], config=model.config)
    torch.cuda.synchronize()
    return c
for S in (1, 64, 256):
    xs = []
    for i in range(15):
        c = with_cache(S); torch.cuda.synchronize(); a = time.perf_counter(); prefill(model, ids[:, 256:256 + S], c, 256); torch.cuda.synchronize(); xs.append((time.perf_counter() - a) * 1e3)
    print("suffix S=%d L=256" % S, round(statistics.median(xs[3:]), 2))
# plain model call without explicit mask/positions, no cache
with torch.inference_mode():
    print("raw model(ids[:257]) use_cache=False", t(lambda: model(input_ids=ids[:, :257], use_cache=False, logits_to_keep=1)))
    print("raw model(ids[:257]) use_cache=True no mask", t(lambda: model(input_ids=ids[:, :257], use_cache=True, logits_to_keep=1)))
    am = torch.ones((1, 257), dtype=torch.long, device="cuda")
    print("raw model(ids[:257]) with all-ones mask", t(lambda: model(input_ids=ids[:, :257], attention_mask=am, use_cache=False, logits_to_keep=1)))
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    prefill(model, ids[:, :257], None, 0); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=18))
