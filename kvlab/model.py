"""Model loading and explicit-position forward helpers (HF reference harness).

Importing this module points HF_HOME at the project-local .hf_cache/ unless the
caller already set HF_HOME, so weights never land in a global cache by default.
It must be imported before anything else imports transformers/huggingface_hub.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DTYPE = torch.bfloat16  # checkpoint's native dtype; see docs/decisions.md
ATTN_IMPLEMENTATION = "sdpa"


def load_model(device: str = "cuda", dtype: torch.dtype = DTYPE):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation=ATTN_IMPLEMENTATION
    )
    model = model.to(device).eval()
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != dtype:
        raise RuntimeError(f"model loaded as {actual_dtype}, expected {dtype}")
    if model.config._attn_implementation != ATTN_IMPLEMENTATION:
        raise RuntimeError(f"attention backend is {model.config._attn_implementation}")
    return model, tok


def model_info(model) -> dict:
    cfg = model.config
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": cfg._attn_implementation,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        "layer_types": sorted(set(cfg.layer_types)),
        "rope_parameters": getattr(cfg, "rope_parameters", None),
        "vocab_size": cfg.vocab_size,
    }


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, cache: DynamicCache | None, start_pos: int):
    """Run one forward over `input_ids` (shape [1, n]) at absolute positions
    [start_pos, start_pos + n), appending to `cache` (mutated in place).

    Positions, the 2D attention mask and the cache length are passed/checked
    explicitly so they cannot silently disagree.
    Returns (last-position logits as float32 [vocab], cache).
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids [1, n], got {tuple(input_ids.shape)}")
    n = input_ids.shape[1]
    if n < 1:
        raise ValueError("prefill needs at least one token")
    if cache is None:
        if start_pos != 0:
            raise ValueError("start_pos must be 0 without a cache")
        cache = DynamicCache(config=model.config)
    if cache.get_seq_length() != start_pos:
        raise ValueError(f"cache length {cache.get_seq_length()} != start_pos {start_pos}")

    device = input_ids.device
    position_ids = torch.arange(start_pos, start_pos + n, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, start_pos + n), dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    if out.past_key_values is not cache:
        raise RuntimeError("model returned a different cache object than it was given")
    if cache.get_seq_length() != start_pos + n:
        raise RuntimeError(f"cache length {cache.get_seq_length()} != {start_pos + n} after forward")
    return out.logits[0, -1].float(), cache


@torch.inference_mode()
def greedy_continue(model, cache: DynamicCache, first_logits: torch.Tensor, start_pos: int, steps: int):
    """Greedy decode `steps` tokens. The first token is argmax(first_logits);
    each later token comes from feeding the previous one at the next position.
    Returns (tokens, per-step float32 logits on CPU)."""
    tokens, step_logits = [], [first_logits.cpu()]
    logits, pos = first_logits, start_pos
    device = first_logits.device
    for i in range(steps):
        tok_id = int(torch.argmax(logits))
        tokens.append(tok_id)
        if i == steps - 1:
            break
        ids = torch.tensor([[tok_id]], device=device)
        logits, cache = prefill(model, ids, cache, start_pos=pos)
        step_logits.append(logits.cpu())
        pos += 1
    return tokens, step_logits
