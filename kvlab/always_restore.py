"""Synchronous always-restore request flow (docs/design.md §8).

M5a (docs/M5_DESIGN.md): an optional, keyword-only restore-length policy. With
policy=None (the default) the request path is the M1 one, unchanged. With a policy,
the engine plans (side-effect free), lets the policy choose a candidate length,
validates it, and restores exactly that prefix through TieredKVCache.restore().
"""

from dataclasses import dataclass

import torch

from kvlab.boundaries import max_restore_len
from kvlab.identity import chain_keys
from kvlab.model import greedy_continue, prefill
from kvlab.restore_policy import RestoreDecision, plan_restore, policy_name, validate_selection
from kvlab.tiered_cache import TieredKVCache


@dataclass
class RequestResult:
    prompt_len: int
    eligible_restore_len: int     # largest L allowed by the rule (L % C == 0, L <= N-1)
    restore_len: int              # actually restored
    restore_used: bool
    sources: list                 # tier per restored chunk
    miss_reason: str | None
    stored_chunks: list           # chunk indices newly stored by this request
    tokens: list
    step_logits: list             # float32 CPU tensors, one per greedy step
    decision: RestoreDecision | None = None  # M5a: set only when a policy is attached and restore is allowed


@dataclass
class Prefilled:
    """State after everything that precedes first-token selection (see prefill_request)."""
    logits: torch.Tensor          # last-position float32 logits, on the model device
    cache: object                 # DynamicCache holding exactly the N prompt tokens
    prompt_len: int
    eligible_restore_len: int
    restore_len: int
    sources: list
    miss_reason: str | None
    stored_chunks: list
    decision: RestoreDecision | None = None  # M5a: set only when a policy is attached and restore is allowed


def _no_mark(phase: str) -> None:
    pass


class AlwaysRestoreEngine:
    def __init__(self, model, tiered: TieredKVCache, *, policy=None):
        self.model, self.tiered, self.C = model, tiered, tiered.C
        self.device = next(model.parameters()).device
        self.policy = policy  # None: the M1 always-restore path; may be replaced between requests

    def prefill_request(self, token_ids, allow_restore: bool = True, on_restored=None, mark=None) -> Prefilled:
        """Everything before the first token is selected, in run() order: ingest (ids to
        device, chain keys), restore, prefill, store. `allow_restore=False` skips the
        restore lookup; it exists only to measure full recomputation offline, and it never
        consults a policy. run() always allows restoration: with policy=None it restores the
        longest valid prefix, and with an attached policy that policy may still choose L = 0
        (recomputation). `mark(phase)` is called after each phase (diagnostic timing only);
        with a policy, "decide" comes between "ingest" and "restore"."""
        mark = mark or _no_mark
        token_ids = list(token_ids)
        n = len(token_ids)
        ids = torch.tensor([token_ids], device=self.device)
        keys = chain_keys(self.tiered.ns_digest, token_ids, self.C)
        l_eligible = max_restore_len(n, self.C)
        mark("ingest")
        decision = None
        if allow_restore and self.policy is not None:
            cache, L, sources, miss_reason, decision = self._restore_with_policy(token_ids, keys, n, mark)
        elif allow_restore:
            cache, L, sources, miss_reason = self.tiered.restore(token_ids, keys[:l_eligible // self.C],
                                                                 self.device, self.model.config)
        else:
            cache, L, sources, miss_reason = None, 0, [], "restore_not_attempted"
        mark("restore")
        c = self.tiered.counters
        c.requests += 1
        if L:
            c.restored_requests += 1
            if on_restored is not None:
                on_restored(cache, L)
            logits, cache = prefill(self.model, ids[:, L:], cache, L)
        else:
            c.full_recompute_requests += 1
            logits, cache = prefill(self.model, ids, None, 0)
        mark("prefill")
        stored = self.tiered.store(cache, token_ids, keys)  # cache holds exactly the N prompt tokens here
        mark("store")
        return Prefilled(logits, cache, n, l_eligible, L, sources, miss_reason, stored, decision)

    def _restore_with_policy(self, token_ids, keys, n, mark):
        """Plan, choose, validate, then restore exactly the chosen prefix [0, selected).
        Validation failures inside it fall back to the verified prefix that restore() returns
        (no re-plan, no second attempt). L = 0 skips restore() entirely: that is the policy's
        choice, not a miss, so no miss counter or miss reason is recorded for it."""
        plan = plan_restore(self.tiered, keys, n)
        choice = self.policy.choose(plan)
        selected = validate_selection(plan, choice)  # raises PolicyError before any restore work
        mark("decide")
        if selected:
            cache, L, sources, miss_reason = self.tiered.restore(token_ids, keys[:selected // self.C],
                                                                 self.device, self.model.config)
        else:
            self.tiered.last_restore = None  # no restore ran: never report an earlier request's restore
            cache, L, sources, miss_reason = None, 0, [], None
        decision = RestoreDecision(policy=policy_name(self.policy), plan=plan, selected_len=selected,
                                   reason=choice.reason, objective_ms=choice.objective_ms, actual_len=L,
                                   actual_sources=tuple(sources),
                                   restore_miss_reason=miss_reason if selected else None)
        return cache, L, sources, miss_reason, decision

    def run(self, token_ids, steps: int = 8, on_restored=None) -> RequestResult:
        p = self.prefill_request(token_ids, True, on_restored)
        tokens, step_logits = greedy_continue(self.model, p.cache, p.logits, p.prompt_len, steps)
        p.cache = None  # the GPU cache is dropped; later requests can only reuse via the tiered store
        return RequestResult(p.prompt_len, p.eligible_restore_len, p.restore_len, p.restore_len > 0, p.sources,
                             p.miss_reason, p.stored_chunks, tokens, step_logits, p.decision)
