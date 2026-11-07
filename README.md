# kv-cache-lab: a tiered prefix KV cache for LLM inference

A prefix KV cache for Hugging Face Transformers inference that keeps attention K/V state in a bounded
pinned-host tier and a bounded disk tier, restores the longest verified prefix of a new request, and
recomputes only the rest. Restoration is checked bitwise against a GPU-resident replay. Each
performance experiment's protocol was fixed before its measurements, and the formal evaluations (M3_CONFIRM_V1,
M4, M5) were analysed by validators that fail closed on incomplete evidence.

```
 request tokens ──► chain keys: key_i = SHA-256(namespace ‖ key_{i-1} ‖ i ‖ tokens of chunk i)
                         │
                         ▼
   ┌───────────── longest contiguous run of present, valid chunks ──────────────┐
   │  pinned host slab (LRU, leases)  ──spill on eviction──►  disk tier (LRU)   │
   │  slots read directly                 CRC-checked files read via staging    │
   └──────────────┬──────────────────────────────────────────────────────────────┘
                  │ host-to-device copies: per layer (baseline, default) or one per chunk (batched)
                  ▼
        GPU: restored prefix [0, L)  +  prefill of [L, N)  ──►  first token, decode
                  │
                  └──► store every new complete prompt chunk back into the host tier
```

## Key results
One RTX 4080 Laptop GPU under WSL2, Qwen2.5-1.5B-Instruct in bf16, 256-token chunks, batch 1,
synchronous first-token latency with a warm page cache for the disk tier. These are restoration and
first-token measurements, not serving TTFT, decode throughput, cold-disk or tail-latency results.

| Experiment | Result |
|---|---|
| **Batched restore** (preregistered confirmation, 8,640 samples) | Replacing 56 per-layer copies per chunk with one contiguous host-to-device copy cut median first-token latency **1.46–1.63×** (ratio of medians) on 4- and 5-chunk prefixes from host and warm disk; all 8 preregistered verdicts "batched faster". In the earlier exploratory M3 profiler session, a 16-chunk restore issued 16 instead of 896 copies for the same bytes. |
| **Request traces** (10 sessions, 11,200 strictly checked requests) | With batched restore the tiered cache was faster than no caching on shared-prompt, multi-turn and document-reuse traces (−37.9, −16.4, −20.1 ms per request) and faster than a half-budget GPU-only prefix cache on the first two (−8.7, −15.1 ms). It was slower than that GPU cache on document reuse (+8.8 ms) and slower than no caching when nothing is reused (+60.5 ms: nothing to restore, every chunk still admitted). With the per-layer baseline backend, 5 of its 7 comparisons were inconclusive. |
| **Cost-aware restore length** (held-out evaluation, 8 cases) | No incremental benefit demonstrated: the calibrated cost model chose exactly what always-restore chose on every case. Its predictions had 5.0% mean absolute relative error over 47 candidates, up to 31% on single candidates. |

Details, intervals, statistical assumptions and the full verdict tables: [docs/results.md](docs/results.md).

## Correctness
- **Storage is verified strictly.** Restored K/V bytes must equal what was written, and every step's
  logits must be bitwise equal to a GPU-resident replay of the same request with the same restore
  length. Both restore backends are also bitwise equal to each other. The trace evaluation compared
  11,200 requests this way with 0 mismatches.
- **Faults degrade to misses, never to wrong output or repeated inference.** Corrupt, truncated,
  missing or foreign chunk files are rejected by identity checks and CRC32 before use; restoration
  stops at the last verified chunk; a full disk skips the cache write. Every request makes exactly one
  prefill and `steps − 1` decode calls. Mutation checks confirm that the tests catch deliberately
  broken copies of the code.
- **Not claimed: equality with full prefill.** In bf16, prefilling in chunks rounds differently from one
  unchunked prefill, so an exact tie can flip a greedy token. One such case (`nat_long`) is kept as a
  recorded FAIL of the original M0 suite, with its criteria unchanged ([D10](docs/decisions.md)).
- **Evidence validation fails closed.** The first M3 analysis script could report a positive verdict
  from incomplete evidence ([VAL-006](docs/BUGLOG.md)). The corrected M3 analysis and every later
  formal analysis (M3_CONFIRM_V1, M4, M5) validate the evidence against a frozen specification first
  and refuse to produce a verdict otherwise.

## Design highlights
- **Identity:** chunk keys chain SHA-256 over a canonical namespace (model, revision, dtype, layout,
  attention backend, library versions) and the exact token ids, so equal tokens after a different
  prefix or under a different model never alias.
- **Chunk files:** a little-endian header with its own CRC32, the raw K/V bytes and a payload CRC32;
  every field is validated before the payload is read, and no allocation is sized by a header field.
- **Budgets and leases:** the host slab is allocated and pinned once; LRU eviction never touches a
  leased chunk; disk bytes are released only after a file is really gone.
- **Ownership:** the disk tier refuses any directory it does not own and never follows symlinks.
- **Two restore backends:** the default `baseline` makes 2 × num_layers per-layer copies per chunk;
  the opt-in `batched` backend makes one contiguous copy per chunk into bounded device staging (at most
  4 chunks, 29.4 MB for this model), then one placement per group. The default stays `baseline`
  ([D30, D31](docs/decisions.md)); the results above name the backend each experiment used.
- **Optional restore-length policies:** always restore, recompute, a fixed length, or the
  cost-model-driven choice evaluated above.

More: [docs/design.md](docs/design.md).

## Quick start
Linux and Python 3.12 or newer (developed with 3.14). The pinned versions are in `requirements.txt`.

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130
pip install transformers==5.22.1 pytest==9.2.0
python -m pytest -q        # CPU suite: tiny random Qwen2 models, no GPU and no model download
```

The GPU suites need CUDA and the pinned model revision, downloaded into the project-local cache:

```bash
HF_HOME=$PWD/.hf_cache python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
HF_HUB_OFFLINE=1 python -m pytest -m "gpu and m1" -v      # tiered cache, both restore backends
HF_HUB_OFFLINE=1 python -m pytest -m "gpu and m3" -v      # restore backends
HF_HUB_OFFLINE=1 python -m pytest -m "gpu and m5" -v      # restore-length policies
HF_HUB_OFFLINE=1 python -m pytest -m "gpu and m0_1" -v    # restored-vs-resident audit
HF_HUB_OFFLINE=1 python -m pytest -m "gpu and m0_strict" -v  # original M0 suite: 1 known FAIL (nat_long)
HF_HUB_OFFLINE=1 python scripts/m4_gate.py --out-dir <new dir>  # M1 + M3 + M4 trace-replay gate
```

`kvlab/model.py` uses `HF_HOME=.hf_cache/` unless `HF_HOME` is set. The M4 trace-replay suite writes
required diagnostics into the gate's output directory, so it runs only through `scripts/m4_gate.py`.

## Reproducing the analyses
The analyses run on CPU from the archived evidence, with the analysis code of the revision that
produced it. The original analysis outputs are committed next to the raw data.

```bash
# M3_CONFIRM_V1: the bound analysis files are unchanged, so the frozen analyzer runs in this tree
python scripts/m3_analyze.py --spec M3_CONFIRM_V1 --run-dir results/m3_confirm/M3_CONFIRM_V1_run \
  --out-dir <new dir> --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216

# M4_EVAL_V2: likewise
python scripts/m4_analyze.py --spec M4_EVAL_V2 --run-dir results/m4/M4_EVAL_V2_run --out-dir <new dir> \
  --expect-freeze-sha256 150f232ef11040ef1bcaf095a3f9beb341e708c59fc91de244f78e848d4cc83a

# M5: run the recorded revision's own scripts against the archived evidence
(cd results/m5 && sha256sum -c SHA256SUMS)
mkdir <src> && git archive a3c63419b6504b20ce419eff62fea2c64f26fb3e | tar -x -C <src>
(cd <src> && python scripts/m5_analyze.py --eval-run-dir <repo>/results/m5/M5_EVAL_V1_run \
  --fit-dir <repo>/results/m5/M5_FIT_V1 --out-dir <new dir>)
cmp <new dir>/analysis.json results/m5/M5_EVAL_V1_analysis/analysis.json
```

The analyzers need the pinned package versions, refuse to overwrite outputs, and exit 3 on any
missing or altered evidence. Rebuilding a frozen source tree for the other experiments, and what
offline reproduction does not verify: [docs/results.md](docs/results.md#reproducing).

## Limitations
- One model and configuration (Qwen2.5-1.5B-Instruct, bf16, SDPA, full attention), batch 1, one
  process, sequential requests. No concurrency, no asynchronous copies or overlap with compute.
- Measurements come from one laptop GPU under WSL2, where restore latency also shows an unexplained
  slow mode of more expensive CUDA calls; the disk tier was measured only with a warm page cache.
- The disk tier is a cache, not storage: no fsync, no recovery across restarts, CRC32 is not
  authentication, and the directory must be trusted.
- Restored output equals a resident replay, not full-prefill output (see Correctness).
- The trace evaluation uses synthetic traces and compares against this project's own GPU-only prefix
  cache, not vLLM or SGLang, with unequal total resources (the tiered cache uses pinned host memory,
  disk and the page cache instead of GPU memory).
- Not integrated with a serving engine.

## Repository layout
```
kvlab/            cache library (identity, chunk_format, host_store, disk_store, tiered_cache,
                  always_restore, restore_policy, restore_cost) and evaluation tooling
                  (benchmarks, trace replay, validators, statistics; docs/README.md has the map)
tests/            CPU tests (default); tests/gpu/ real-model tests (-m gpu)
scripts/          experiment runners, analyzers, gates and reconstruction tools
results/          raw evidence and analysis outputs, per experiment
docs/             design, results, decisions, bug log, correctness matrix, experiment protocols
```

Documentation index: [docs/README.md](docs/README.md).

## License
MIT; see [LICENSE](LICENSE). Model weights are not part of this repository; they are downloaded from
Hugging Face under their own license.
