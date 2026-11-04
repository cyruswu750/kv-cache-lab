"""GPU-resident replay used to verify the tiered store strictly.

It replays the same request sequence with the same restore lengths and the
same newly stored chunks the tiered engine reported, but keeps chunk K/V as
GPU tensors (never copied to CPU or disk). Where the tiered store is lossless,
every step logit must be bitwise equal and every restored chunk byte-identical.
"""

import torch
from transformers import DynamicCache

from kvlab.identity import chain_keys
from kvlab.kv_snapshot import bytes_equal
from kvlab.model import greedy_continue, prefill


class ResidentReference:
    def __init__(self, model, ns_digest: bytes, chunk_size: int):
        self.model, self.ns_digest, self.C = model, ns_digest, chunk_size
        self.device = next(model.parameters()).device
        self.chunks: dict[bytes, list] = {}  # key -> [(k, v)] per layer, GPU

    def check_restored(self, token_ids, cache, L) -> dict:
        """Byte-compare a restored cache prefix [0, L) with the resident chunks."""
        keys = chain_keys(self.ns_digest, token_ids, self.C)
        checked = mismatched = 0
        for i in range(L // self.C):
            ref = self.chunks[keys[i]]
            for li, layer in enumerate(cache.layers):
                s, e = i * self.C, (i + 1) * self.C
                for got, want in ((layer.keys[:, :, s:e, :], ref[li][0]), (layer.values[:, :, s:e, :], ref[li][1])):
                    checked += 1
                    mismatched += not bytes_equal(got, want)
        return {"tensors_checked": checked, "mismatched": mismatched, "all_bytes_equal": checked > 0 and not mismatched}

    def run(self, token_ids, restore_len: int, stored_chunks, steps: int = 8):
        token_ids = list(token_ids)
        ids = torch.tensor([token_ids], device=self.device)
        keys = chain_keys(self.ns_digest, token_ids, self.C)
        if restore_len:
            n = restore_len // self.C
            data = [(torch.cat([self.chunks[keys[i]][li][0] for i in range(n)], dim=-2),
                     torch.cat([self.chunks[keys[i]][li][1] for i in range(n)], dim=-2))
                    for li in range(self.model.config.num_hidden_layers)]
            cache = DynamicCache(ddp_cache_data=data, config=self.model.config)
            logits, cache = prefill(self.model, ids[:, restore_len:], cache, restore_len)
        else:
            logits, cache = prefill(self.model, ids, None, 0)
        for i in stored_chunks:  # mirror exactly what the tiered engine stored (same provenance)
            s, e = i * self.C, (i + 1) * self.C
            self.chunks[keys[i]] = [(layer.keys[:, :, s:e, :].clone(), layer.values[:, :, s:e, :].clone())
                                    for layer in cache.layers]
        tokens, step_logits = greedy_continue(self.model, cache, logits, len(token_ids), steps)
        del cache
        return tokens, step_logits
