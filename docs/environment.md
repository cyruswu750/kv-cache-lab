# Environment (observed 2026-11-02)

Machine-readable copy: `results/m0/environment.json` (from `scripts/env_report.py`).
Full package list: `results/m0/pip_freeze.txt`.

| Item | Observed |
|---|---|
| OS | Ubuntu 26.04.1 LTS on WSL2, kernel `6.18.54.1-microsoft-standard-WSL2` |
| CPU / RAM | 32 logical CPUs; 7.4 GiB RAM visible to WSL (MemTotal 7720152 kB), 2 GiB swap |
| Disk | `/` (ext4 VHD): 1007 G total, ~940 GiB free |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9, 12282 MiB total (≈11.0 GiB free to PyTorch at start) |
| Driver | Windows host driver 596.49 (`nvidia-smi` reports NVIDIA-SMI 595.71.05, max CUDA 13.2) |
| System CUDA toolkit | none (`/usr/local/cuda*` absent); not needed, the CUDA runtime comes from the pip wheels |
| System Python | 3.14.4 (`/usr/bin/python3`), **no `ensurepip`/python3-venv package**; no system pip, uv, or conda |
| Project env | `.venv/` created with `python3 -m venv --without-pip` + official `get-pip.py` (pip 26.2.2) |
| PyTorch | `torch==2.14.1+cu130` from `https://download.pytorch.org/whl/cu130`; CUDA runtime 13.0, cuDNN 9.24.0 |
| CUDA from PyTorch | works: `torch.cuda.is_available()=True`, bf16 matmul OK (`results/m0/logs/cuda_check.log`) |
| Transformers | 5.22.1 (tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.4.0) |
| Test runner | pytest 9.2.0 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, `model.safetensors` 3,087,467,144 B |
| Model storage | `HF_HOME=.hf_cache/` (project-local, git-ignored), 2.9 GiB |
| Actual dtype | `torch.bfloat16` (checked after load) |
| Attention backend | `sdpa` (requested explicitly, checked after load) |
| Model config | 28 layers, 12 Q heads, 2 KV heads, head_dim 128, all `full_attention`, `use_sliding_window=false`, rope_theta 1e6 |
| KV size | 28 layers × 2 (K,V) × 2 heads × 128 × 2 B = **28 KiB/token** (measured: 512 tokens = 14,680,064 B) |

## Cache API observed in transformers 5.22.1
- The model returns a `DynamicCache`. `cache.layers` is a list of 28 `DynamicLayer`.
  Each layer has `.keys` / `.values` shaped `[batch, kv_heads, seq, head_dim]`, contiguous.
- `DynamicCache(ddp_cache_data=[(k, v), ...], config=model.config)` builds a
  pre-filled cache. `DynamicLayer.update` does `torch.cat`, so the cache
  copies the data it is given.
- `Qwen2Model.forward` has no `cache_position` argument. If `position_ids` is
  omitted, it uses `arange(n) + past_key_values.get_seq_length()`. The 2D
  `attention_mask` must cover `past + new` tokens.
- `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`, not a list.

## Nothing modified outside the project
No sudo was used. System packages, the driver, the global Python, and shell
startup files were not changed. Weights are not stored in `~/.cache`. The
only file outside the project is `get-pip.py`, downloaded to the session
scratch directory.

## Further observations (2026-11-03)
- Re-checked: same torch/transformers/driver/GPU, and the GPU was idle at start.
- Pinned host memory: the installed torch's `CachingHostAllocator.h` rounds
  `pin_memory=True` allocations up to a power of two (below
  `pinned_max_round_threshold`) and caches freed blocks.
  `torch.cuda.host_memory_stats()` returned an empty dict here, so allocator
  usage could not be observed through it.
- `torch.cuda.cudart().cudaHostRegister` / `cudaHostUnregister` work on this
  WSL2 setup: `is_pinned()` becomes True and then False again. M1 uses them
  for one exact-size slab (decisions D16).
- Disk tier files live under `.kvcache/` (git-ignored) for scripts, or under
  pytest `tmp_path` for tests. They sit on the WSL2 ext4 VHD, and reads may be
  served by the Linux page cache and by the Windows host cache underneath it.
