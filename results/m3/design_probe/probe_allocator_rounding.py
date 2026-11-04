"""Exploratory check (M3 review): does memory_allocated exceed requested bytes for the restored
cache's per-layer tensor sizes? Synthetic, no model: 56 tensors via the same torch.cat path as
DynamicCache (empty + x), for the per-layer K/V sizes of L = 1024, 1536, 3072, 4096."""
import json
import torch
from transformers import DynamicCache
out = {}
for L in (1024, 1536, 3072, 4096):
    src = torch.empty((28, 2, 1, 2, L, 128), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    st0 = torch.cuda.memory_stats()
    c = DynamicCache(ddp_cache_data=[(src[l, 0], src[l, 1]) for l in range(28)])
    torch.cuda.synchronize()
    st1 = torch.cuda.memory_stats()
    out[L] = {"tensor_bytes_each": 2 * L * 128 * 2, "payload_bytes_56": 56 * 2 * L * 128 * 2,
              "allocated_bytes_delta": st1["allocated_bytes.all.current"] - st0["allocated_bytes.all.current"],
              "requested_bytes_delta": st1["requested_bytes.all.current"] - st0["requested_bytes.all.current"]}
    del c, src
    torch.cuda.empty_cache()
print(json.dumps(out, indent=1))
