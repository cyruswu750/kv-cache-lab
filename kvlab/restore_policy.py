"""M5a: side-effect-free restore planning and opt-in restore-length policies.

Design: docs/M5_DESIGN.md. Nothing here is used unless a policy is passed to
`AlwaysRestoreEngine(..., policy=...)`; with policy=None the engine keeps the M1 path.

Planning (`plan_restore`) inspects only tier membership (`TieredKVCache.tier_of`) and
configuration. It never reads or validates a chunk file, leases or touches an entry,
changes LRU order, counters, slots, staging state, files or `last_restore`, allocates
restore tensors, or runs the model. A planned chunk is *indexed*, not validated: its
payload is checked only when `TieredKVCache.restore()` reads it.

Candidate restore lengths for prompt length N and chunk size C are
  0, C, 2C, ..., k*C
where k*C <= max_restore_len(N, C) (first-token rule, docs/design.md §2) and k stops at
  - the first chunk that is not indexed in any tier ("absent"), or
  - the first disk chunk when the host store has no staging buffer, since restore() can
    never read it ("no_staging_buffer").
L = 0 (full recomputation) is always a candidate.

A policy maps a plan to a `PolicyDecision`. The engine validates the decision with
`validate_selection` before any restore runs; invalid outputs raise `PolicyError` and are
never clamped.
"""

import math
import numbers
from dataclasses import dataclass

from kvlab.boundaries import max_restore_len

TIERS = ("host", "disk")
LIMIT_REASONS = ("no_eligible_chunk", "eligible_limit", "absent", "no_staging_buffer")
PRESENCE = "indexed_not_validated"  # what a planned chunk is known to be


class PolicyError(ValueError):
    """A policy returned an invalid decision, or a forced selection is not a feasible candidate."""


@dataclass(frozen=True)
class RestorePlan:
    prompt_len: int
    chunk_size: int
    eligible_len: int                 # max_restore_len(N, C): the first-token rule's largest L
    indexed_tiers: tuple              # tier of each chunk of the contiguous indexed prefix, within eligible_len
    feasible_lengths: tuple           # candidates, ascending: (0, C, ..., max_feasible_len)
    limit_reason: str                 # why the candidates stop (LIMIT_REASONS)
    namespace_digest: str             # hex; for model compatibility checks
    restore_backend: str
    restore_staging_chunks: int
    disk_staging_available: bool

    @property
    def max_feasible_len(self) -> int:
        return self.feasible_lengths[-1]

    @property
    def feasible_tiers(self) -> tuple:
        """Tiers of the chunks a maximal feasible restore would read, in order."""
        return self.indexed_tiers[:self.max_feasible_len // self.chunk_size]

    def to_dict(self) -> dict:
        return {"prompt_len": self.prompt_len, "chunk_size": self.chunk_size, "eligible_len": self.eligible_len,
                "indexed_tiers": list(self.indexed_tiers), "feasible_lengths": list(self.feasible_lengths),
                "limit_reason": self.limit_reason, "presence": PRESENCE,
                "namespace_digest": self.namespace_digest, "restore_backend": self.restore_backend,
                "restore_staging_chunks": self.restore_staging_chunks,
                "disk_staging_available": self.disk_staging_available}


def plan_restore(tiered, keys, prompt_len: int) -> RestorePlan:
    """Plan the candidate restore lengths of one request (no side effects; see module doc).
    `keys` are the chain keys of every complete prompt chunk (len == prompt_len // C)."""
    C = tiered.C
    if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len < 0:
        raise ValueError(f"prompt_len must be an int >= 0, got {prompt_len!r}")
    if len(keys) != prompt_len // C:
        raise ValueError(f"{len(keys)} chunk keys for prompt length {prompt_len} (chunk {C}); "
                         f"expected {prompt_len // C}")
    tiered._check_backend()  # the configuration restore() would reject is rejected before choosing
    eligible = max_restore_len(prompt_len, C)
    n_eligible = eligible // C
    tiers = []
    for key in keys[:n_eligible]:
        tier = tiered.tier_of(key)  # dict membership only: no lease, no LRU touch, no file access
        if tier is None:
            break
        tiers.append(tier)
    staging = tiered.host.staging_slots > 0
    feasible = len(tiers)
    if not staging and "disk" in tiers:
        feasible = tiers.index("disk")
    if n_eligible == 0:
        reason = "no_eligible_chunk"
    elif feasible < len(tiers):
        reason = "no_staging_buffer"
    elif len(tiers) < n_eligible:
        reason = "absent"
    else:
        reason = "eligible_limit"
    return RestorePlan(prompt_len=prompt_len, chunk_size=C, eligible_len=eligible, indexed_tiers=tuple(tiers),
                       feasible_lengths=tuple(range(0, feasible * C + 1, C)), limit_reason=reason,
                       namespace_digest=tiered.ns_digest.hex(), restore_backend=tiered.restore_backend,
                       restore_staging_chunks=tiered.restore_staging_chunks, disk_staging_available=staging)


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's output. Not validated on construction: the engine validates it against the plan."""
    selected_len: int
    reason: str
    objective_ms: tuple | None = None  # ((L, predicted objective ms), ...) when a cost model was consulted


def validate_selection(plan: RestorePlan, decision) -> int:
    """The selected length as an int, or PolicyError. Rejects non-decisions, booleans, non-integral
    values, negative or non-chunk-aligned lengths and lengths outside the plan's candidates."""
    if not isinstance(decision, PolicyDecision):
        raise PolicyError(f"policy returned {type(decision).__name__}, expected PolicyDecision")
    if not isinstance(decision.reason, str) or not decision.reason:
        raise PolicyError(f"policy decision needs a non-empty reason, got {decision.reason!r}")
    if decision.objective_ms is not None and not isinstance(decision.objective_ms, tuple):
        raise PolicyError(f"objective_ms must be None or a tuple, got {type(decision.objective_ms).__name__}")
    v = decision.selected_len
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):  # also rejects floats such as 256.0
        raise PolicyError(f"selected length must be an integer, got {v!r} ({type(v).__name__})")
    v = int(v)
    if v < 0:
        raise PolicyError(f"selected length {v} is negative")
    if v % plan.chunk_size:
        raise PolicyError(f"selected length {v} is not a multiple of the chunk size {plan.chunk_size}")
    if v not in plan.feasible_lengths:
        raise PolicyError(f"selected length {v} is not a feasible candidate {list(plan.feasible_lengths)} "
                          f"(limit: {plan.limit_reason})")
    return v


@dataclass(frozen=True)
class RestoreDecision:
    """Immutable per-request record of an opt-in policy decision and its execution."""
    policy: str
    plan: RestorePlan
    selected_len: int                 # what the policy chose (validated)
    reason: str
    objective_ms: tuple | None
    actual_len: int                   # what restore() verified and returned (<= selected_len)
    actual_sources: tuple
    restore_miss_reason: str | None   # restore()'s reason when actual_len < selected_len; None if no restore ran

    @property
    def truncated(self) -> bool:
        return self.actual_len < self.selected_len

    def to_dict(self) -> dict:
        return {"policy": self.policy, "plan": self.plan.to_dict(), "selected_len": self.selected_len,
                "reason": self.reason,
                "objective_ms": None if self.objective_ms is None else [list(x) for x in self.objective_ms],
                "actual_len": self.actual_len, "actual_sources": list(self.actual_sources),
                "restore_miss_reason": self.restore_miss_reason, "truncated": self.truncated}


def policy_name(policy) -> str:
    return getattr(policy, "name", type(policy).__name__)


# ---- policies ------------------------------------------------------------------------------

class AlwaysRestore:
    """The longest feasible prefix: the M1 choice, expressed as an opt-in policy."""
    name = "always_restore"

    def choose(self, plan: RestorePlan) -> PolicyDecision:
        if plan.max_feasible_len == 0:
            return PolicyDecision(0, "no_feasible_restore")
        return PolicyDecision(plan.max_feasible_len, "longest_feasible_prefix")


class Recompute:
    """Always L = 0: full prefill with the cache's admission unchanged (not the no-cache C0 baseline)."""
    name = "recompute"

    def choose(self, plan: RestorePlan) -> PolicyDecision:
        return PolicyDecision(0, "policy_recompute")


class FixedChunks:
    """Exactly `chunks` chunks (L = chunks * C), for forced-candidate measurements and tests.
    If that length is not a feasible candidate it raises PolicyError before any restore: it never
    substitutes another length."""
    name = "fixed_chunks"

    def __init__(self, chunks: int):
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 0:
            raise ValueError(f"chunks must be an int >= 0, got {chunks!r}")
        self.chunks = chunks

    def choose(self, plan: RestorePlan) -> PolicyDecision:
        L = self.chunks * plan.chunk_size
        if L not in plan.feasible_lengths:
            raise PolicyError(f"fixed_chunks={self.chunks} (L={L}) is not feasible: candidates "
                              f"{list(plan.feasible_lengths)}, limit {plan.limit_reason}")
        return PolicyDecision(L, "fixed_chunks")


class CostAware:
    """argmin over the feasible candidates of a cost model's predicted objective.

    - The model must be compatible with the plan (namespace, chunk size, backend, staging
      group); otherwise the model raises (a configuration error, never a fallback).
    - No positive candidate: L = 0 without consulting the model.
    - A plan outside the model's declared calibration domain: the always-restore choice, with
      the reason recorded (the documented fallback; only `domain_violation` decides this).
    - Exact ties go to the larger L, so the policy departs from always-restore only on a strict
      predicted improvement.
    """
    name = "cost_aware"

    def __init__(self, model):
        self.model = model  # a validated kvlab.restore_cost.RestoreCostModel (loaded once)

    def choose(self, plan: RestorePlan) -> PolicyDecision:
        self.model.check_compatible(plan)
        if plan.max_feasible_len == 0:
            return PolicyDecision(0, "no_feasible_restore")
        violation = self.model.domain_violation(plan)
        if violation is not None:
            return PolicyDecision(plan.max_feasible_len, f"out_of_domain_always_restore:{violation}")
        objective = self.model.candidate_objectives(plan)
        best_len, best = None, math.inf
        for L, ms in objective:  # ascending L; "<=" sends exact ties to the larger L
            if not math.isfinite(ms):
                raise RuntimeError(f"non-finite predicted objective {ms!r} at L={L}")
            if ms <= best:
                best_len, best = L, ms
        return PolicyDecision(best_len, "min_predicted_objective", objective)
