"""Design probe (exploratory): which CUDA runtime calls do candidate restore copy patterns issue?
Synthetic tensors with the real layout (28 layers x K/V x [1, 2, 256, 128] bf16); no model."""
import json, sys, collections
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import DynamicCache

Lyr, H, C, D = 28, 2, 256, 128
dev = torch.device("cuda")
CB = Lyr * 2 * H * C * D * 2

def slab(nchunks):
    t = torch.empty(nchunks * CB, dtype=torch.uint8)
    rt = torch.cuda.cudart()
    assert rt.cudaHostRegister(t.data_ptr(), t.numel(), 0) == rt.cudaError.success
    t.view(torch.int16).copy_(torch.randint(-30000, 30000, (t.numel() // 2,), dtype=torch.int16))
    return t

def typed(flat):
    return flat.view(torch.bfloat16).view(Lyr, 2, 1, H, C, D)

def count(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    api, gpu = collections.Counter(), collections.Counter()
    for e in prof.events():
        dt = str(e.device_type)
        if "CUDA" in dt:
            gpu["memcpy" if "Memcpy" in e.name else ("memset" if "Memset" in e.name else "kernel")] += 1
        elif e.name.startswith("cuda"):
            api[e.name] += 1
    api.pop("cudaDeviceSynchronize", None)  # the probe's own final sync
    return {"api": dict(api), "gpu_ops": dict(gpu)}

out = {"torch": torch.__version__}
for n in (1, 4):
    s = slab(n)
    # baseline pattern (kvlab/tiered_cache.py M1): 56 per-layer copies per chunk into a [1,H,n*C,D] slice
    def baseline_copies():
        dest = [(torch.empty((1, H, n * C, D), dtype=torch.bfloat16, device=dev),
                 torch.empty((1, H, n * C, D), dtype=torch.bfloat16, device=dev)) for _ in range(Lyr)]
        for i in range(n):
            src = typed(s[i * CB:(i + 1) * CB])
            for li, (k, v) in enumerate(dest):
                k[:, :, i * C:(i + 1) * C, :].copy_(src[li, 0])
                v[:, :, i * C:(i + 1) * C, :].copy_(src[li, 1])
        return dest
    out[f"baseline_copies_n{n}"] = count(baseline_copies)
    # candidate A: one strided copy_ per chunk into a layer-major destination
    def strided_per_chunk():
        Y = torch.empty((Lyr, 2, 1, H, n * C, D), dtype=torch.bfloat16, device=dev)
        for i in range(n):
            Y[:, :, :, :, i * C:(i + 1) * C, :].copy_(typed(s[i * CB:(i + 1) * CB]))
        return Y
    out[f"strided_copy_per_chunk_n{n}"] = count(strided_per_chunk)
    # candidate B: contiguous H2D per chunk into chunk-major scratch, then one permuted placement
    def chunk_major():
        X = torch.empty((n, CB), dtype=torch.uint8, device=dev)
        for i in range(n):
            X[i].copy_(s[i * CB:(i + 1) * CB])
        Y = torch.empty((Lyr, 2, 1, H, n * C, D), dtype=torch.bfloat16, device=dev)
        Y.unflatten(4, (n, C)).copy_(X.view(torch.bfloat16).view(n, Lyr, 2, 1, H, C, D).permute(1, 2, 3, 4, 0, 5, 6))
        return Y
    out[f"chunk_major_n{n}"] = count(chunk_major)
    # correctness of B vs baseline (bytes)
    d, Y = baseline_copies(), chunk_major()
    out[f"bytes_equal_B_vs_baseline_n{n}"] = all(torch.equal(Y[l, 0].view(torch.int16), d[l][0].view(torch.int16)) and
                                                torch.equal(Y[l, 1].view(torch.int16), d[l][1].view(torch.int16)) for l in range(Lyr))
    Ya = strided_per_chunk()
    out[f"bytes_equal_A_vs_baseline_n{n}"] = all(torch.equal(Ya[l, 0].view(torch.int16), d[l][0].view(torch.int16)) for l in range(Lyr))
    # the DynamicCache public constructor (decisions D4) on layer-major views
    def construct():
        return DynamicCache(ddp_cache_data=[(Y[l, 0], Y[l, 1]) for l in range(Lyr)])
    out[f"dynamic_cache_constructor_n{n}"] = count(construct)
    c = construct()
    out[f"constructor_owns_copy_n{n}"] = all(c.layers[l].keys.data_ptr() != Y[l, 0].data_ptr() for l in range(Lyr))
    rt = torch.cuda.cudart(); rt.cudaHostUnregister(s.data_ptr())
print(json.dumps(out, indent=1))
