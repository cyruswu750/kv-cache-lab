"""M2 diagnostic: which restore component shows the ~50 ms slow mode? (not a model input)

Usage: .venv/bin/python scripts/m2_spike_probe.py --out results/m2/<run>/spike_probe.json
Times each component back to back (wall clock + device sync), 200 times after 10 warmups,
and reports the share of samples slower than 1.3x the component's minimum.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402
import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from kvlab import perf  # noqa: E402
from kvlab.model import prefill  # noqa: E402
from kvlab.perf_harness import Bench, gpu_state, start_gpu_monitor, stop_gpu_monitor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    root = km.PROJECT_ROOT / ".kvcache" / "m2" / "spike_probe"
    model, tok = km.load_model()
    monitor = start_gpu_monitor(out.with_suffix(".gpu_monitor.csv"))
    b = Bench(model, tok, root, max_L=1024)
    b.prepare(256)
    ns, C, dev = b.ns, perf.C, b.device
    key, ident = b.keys[0], b.idents[0]
    dest = [(torch.empty((1, ns.num_kv_heads, C, ns.head_dim), dtype=torch.bfloat16, device=dev),
             torch.empty((1, ns.num_kv_heads, C, ns.head_dim), dtype=torch.bfloat16, device=dev))
            for _ in range(ns.num_layers)]
    pageable = torch.empty(b.chunk_bytes, dtype=torch.uint8)
    with b.host_A.lease(key) as (flat, _):
        pageable.copy_(flat)
    flat_dev = torch.empty(b.chunk_bytes, dtype=torch.uint8, device=dev)

    def h2d_56(src_flat):
        src = b.tiered_A._typed(src_flat)
        for li, (k, v) in enumerate(dest):
            k.copy_(src[li, 0])
            v.copy_(src[li, 1])

    def registered_56():
        with b.host_A.lease(key) as (flat, _):
            h2d_56(flat)

    def registered_single_copy():
        with b.host_A.lease(key) as (flat, _):
            flat_dev.copy_(flat)

    def disk_read():
        with b.host_B.staging() as st:
            assert b.disk_B.read_into(key, ns, ident, st.numpy())[0]

    _, cache256 = prefill(model, torch.tensor([b.base[:256]], device=dev), None, 0)
    kv = [(layer.keys.clone(), layer.values.clone()) for layer in cache256.layers]
    del cache256
    suffix_ids = torch.tensor([b.base[256:257]], device=dev)

    def suffix_prefill():
        c = DynamicCache(ddp_cache_data=kv, config=model.config)
        prefill(model, suffix_ids, c, 256)

    tests = {"h2d_56_copies_from_registered_slab": registered_56,
             "h2d_56_copies_from_pageable": lambda: h2d_56(pageable),
             "h2d_single_7MB_copy_from_registered_slab": registered_single_copy,
             "dynamic_cache_assembly_1_chunk": lambda: DynamicCache(ddp_cache_data=dest, config=model.config),
             "disk_read_into_validated_1_chunk": disk_read,
             "dynamic_cache_plus_suffix_prefill_S1_L256": suffix_prefill,
             "host_restore_path_1_chunk_e2e": lambda: b.measure("host", 1),
             "full_path_N257_e2e": lambda: b.measure("full", 1)}
    report = {"n": args.n, "gpu_state_start": gpu_state(), "components": {}}
    for name, fn in tests.items():
        ts = []
        for i in range(10 + args.n):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            if i >= 10:
                ts.append(t1 - t0)
        s = perf.summarize([perf.ns_to_ms(t) for t in ts])
        report["components"][name] = {"samples_ns": ts, **s}
        print(f"{name:45s} median {s['median_ms']:8.3f}  min {s['min_ms']:8.3f}  max {s['max_ms']:8.2f}  "
              f"slow share {s['slow_share_gt_1p3x_min']:.0%}", flush=True)
    report["gpu_state_end"] = gpu_state()
    b.close()
    stop_gpu_monitor(monitor)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
