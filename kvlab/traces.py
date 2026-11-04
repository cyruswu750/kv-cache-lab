"""Synthetic request traces and their offline statistics (M4-A; CPU only, no model).

The traces are SYNTHETIC. Token ids come from a seeded generator and are not natural language.
They model only the structure of prefix reuse: shared prefixes, growing conversations,
popularity skew and working-set size. They are not production traffic, and nothing about real
traffic follows from them.

Four generators (docs/design.md §12):
- `no_reuse` (T0): unique prompts. The first token of every request is distinct, so no two
  requests share any prefix, whatever the chunk size.
- `shared_prefix` (T1): one shared prefix, then a unique suffix per request.
- `multi_turn` (T2): conversations interleaved turn by turn. Each turn's prompt extends the
  previous turn's prompt by a synthetic answer and a new user turn. The answer tokens are drawn
  by the generator, not produced by a model.
- `doc_reuse` (T3): a shared instruction, then one of several documents drawn with Zipf-like
  popularity, then a unique question.

Determinism: only `random.Random(seed)` is used, never Python's `hash()`. The trace digest is
the SHA-256 of a canonical JSON encoding (`kvlab.identity.canonical_json`), so equal digests
mean identical requests in identical order.

Statistics follow the engine's rules: the first-token rule (`kvlab.boundaries`), the
longest-contiguous-prefix lookup, and storage of every complete prompt chunk. Chunk identity
uses `kvlab.identity.chain_keys` under a fixed statistics namespace digest: within one trace,
chunk keys depend only on the tokens.
"""

import hashlib
import json
import random
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from kvlab.boundaries import max_restore_len
from kvlab.identity import canonical_json, chain_keys

TRACE_FORMAT = "kvlab-trace-v1"
SYNTHETIC_NOTE = ("synthetic token sequences from a seeded generator; they model prefix-reuse structure only, "
                  "not natural language or production traffic")
STATS_NS_DIGEST = hashlib.sha256(b"kvlab-trace-stats-v1").digest()


@dataclass(frozen=True)
class TraceRequest:
    id: str
    group: str
    tokens: tuple


@dataclass(frozen=True)
class Trace:
    header: dict          # format, generator, params, seed, synthetic, note
    requests: tuple       # TraceRequest, in replay order

    @property
    def digest(self) -> str:
        return trace_digest(self.header, self.requests)


def trace_digest(header: dict, requests) -> str:
    body = {"header": header, "requests": [[r.id, r.group, list(r.tokens)] for r in requests]}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _header(generator: str, seed: int, params: dict) -> dict:
    return {"format": TRACE_FORMAT, "generator": generator, "seed": seed, "params": params,
            "synthetic": True, "note": SYNTHETIC_NOTE}


def _check(params: dict, n_markers: int) -> None:
    lo, hi = params["vocab"]
    if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi):
        raise ValueError(f"vocab range {params['vocab']!r} must be [lo, hi) with 0 <= lo < hi")
    if n_markers > hi - lo:
        raise ValueError(f"{n_markers} distinct marker tokens do not fit in vocab range [{lo}, {hi})")
    for k, v in params.items():
        if k != "vocab" and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0):
            raise ValueError(f"parameter {k}={v!r} must be a non-negative number")
    for a, b in (("min_len", "max_len"), ("suffix_min", "suffix_max"), ("user_min", "user_max"),
                 ("answer_min", "answer_max"), ("doc_min", "doc_max"), ("question_min", "question_max")):
        if a in params and not params[a] <= params[b]:
            raise ValueError(f"{a}={params[a]} > {b}={params[b]}")


def _tokens(rng: random.Random, n: int, vocab) -> list:
    lo, hi = vocab
    return [rng.randrange(lo, hi) for _ in range(n)]


def no_reuse(seed: int, *, n_requests: int, min_len: int, max_len: int, vocab=(10, 150000)) -> Trace:
    """T0: unique prompts with lengths uniform in [min_len, max_len] (min_len >= 1). The first token
    of every request is distinct, so the oracle reuse is exactly zero at any chunk size."""
    params = {"n_requests": n_requests, "min_len": min_len, "max_len": max_len, "vocab": list(vocab)}
    _check(params, n_requests)
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    rng = random.Random(seed)
    markers = rng.sample(range(*vocab), n_requests)
    reqs = tuple(TraceRequest(f"r{i}", f"r{i}", tuple([markers[i]] + _tokens(rng, rng.randint(min_len, max_len) - 1, vocab)))
                 for i in range(n_requests))
    return Trace(_header("no_reuse", seed, params), reqs)


def shared_prefix(seed: int, *, n_requests: int, prefix_len: int, suffix_min: int, suffix_max: int,
                  vocab=(10, 150000)) -> Trace:
    """T1: one shared prefix of prefix_len tokens, then a unique suffix (suffix_min >= 1) whose first
    token is distinct per request, so requests share exactly the prefix."""
    params = {"n_requests": n_requests, "prefix_len": prefix_len, "suffix_min": suffix_min, "suffix_max": suffix_max,
              "vocab": list(vocab)}
    _check(params, n_requests)
    if suffix_min < 1:
        raise ValueError("suffix_min must be >= 1")
    rng = random.Random(seed)
    prefix = _tokens(rng, prefix_len, vocab)
    markers = rng.sample(range(*vocab), n_requests)
    reqs = tuple(TraceRequest(f"r{i}", "shared", tuple(prefix + [markers[i]] + _tokens(rng, rng.randint(suffix_min, suffix_max) - 1, vocab)))
                 for i in range(n_requests))
    return Trace(_header("shared_prefix", seed, params), reqs)


def multi_turn(seed: int, *, n_conversations: int, n_turns: int, opening_len: int, user_min: int, user_max: int,
               answer_min: int, answer_max: int, vocab=(10, 150000)) -> Trace:
    """T2: requests in turn-major order (turn 0 of every conversation, then turn 1, ...). The prompt
    of turn t+1 is the prompt of turn t + a synthetic answer + a new user turn. Each conversation
    opens with a distinct marker token (opening_len >= 1), so conversations share nothing."""
    params = {"n_conversations": n_conversations, "n_turns": n_turns, "opening_len": opening_len,
              "user_min": user_min, "user_max": user_max, "answer_min": answer_min, "answer_max": answer_max,
              "vocab": list(vocab)}
    _check(params, n_conversations)
    if opening_len < 1 or user_min < 1:
        raise ValueError("opening_len and user_min must be >= 1")
    rng = random.Random(seed)
    markers = rng.sample(range(*vocab), n_conversations)
    history = [[markers[c]] + _tokens(rng, opening_len - 1, vocab) for c in range(n_conversations)]
    reqs = []
    for t in range(n_turns):
        for c in range(n_conversations):
            if t > 0:
                history[c] += _tokens(rng, rng.randint(answer_min, answer_max), vocab)  # synthetic answer
            history[c] += _tokens(rng, rng.randint(user_min, user_max), vocab)          # user turn
            reqs.append(TraceRequest(f"c{c}t{t}", f"c{c}", tuple(history[c])))
    return Trace(_header("multi_turn", seed, params), tuple(reqs))


def doc_reuse(seed: int, *, n_requests: int, n_docs: int, instruction_len: int, doc_min: int, doc_max: int,
              question_min: int, question_max: int, zipf_s: float, vocab=(10, 150000)) -> Trace:
    """T3: request = shared instruction + document + question. Documents are drawn with weight
    1 / (rank + 1) ** zipf_s. Each document and each question starts with a distinct marker token
    (doc_min, question_min >= 1), so two requests share the instruction, plus the whole document
    when it is the same one."""
    params = {"n_requests": n_requests, "n_docs": n_docs, "instruction_len": instruction_len, "doc_min": doc_min,
              "doc_max": doc_max, "question_min": question_min, "question_max": question_max, "zipf_s": zipf_s,
              "vocab": list(vocab)}
    _check(params, n_docs + n_requests)
    if doc_min < 1 or question_min < 1 or n_docs < 1:
        raise ValueError("doc_min, question_min and n_docs must be >= 1")
    rng = random.Random(seed)
    instruction = _tokens(rng, instruction_len, vocab)
    markers = rng.sample(range(*vocab), n_docs + n_requests)
    docs = [[markers[d]] + _tokens(rng, rng.randint(doc_min, doc_max) - 1, vocab) for d in range(n_docs)]
    weights = [1.0 / (d + 1) ** zipf_s for d in range(n_docs)]
    reqs = []
    for i in range(n_requests):
        d = rng.choices(range(n_docs), weights=weights)[0]
        question = [markers[n_docs + i]] + _tokens(rng, rng.randint(question_min, question_max) - 1, vocab)
        reqs.append(TraceRequest(f"r{i}", f"doc{d}", tuple(instruction + docs[d] + question)))
    return Trace(_header("doc_reuse", seed, params), tuple(reqs))


GENERATORS = {"no_reuse": no_reuse, "shared_prefix": shared_prefix, "multi_turn": multi_turn, "doc_reuse": doc_reuse}


# ---- files
def dump(trace: Trace, path) -> str:
    """Write a JSONL trace (header line with the digest, then one request per line). Never
    overwrites. Returns the digest."""
    digest = trace.digest
    with open(path, "x") as f:
        f.write(json.dumps({**trace.header, "trace_sha256": digest, "n_requests": len(trace.requests)}) + "\n")
        for r in trace.requests:
            f.write(json.dumps({"id": r.id, "group": r.group, "tokens": list(r.tokens)}) + "\n")
    return digest


def load(path) -> Trace:
    """Read a JSONL trace and check its recorded digest and request count."""
    lines = Path(path).read_text().splitlines()
    head = json.loads(lines[0])
    want, n = head.pop("trace_sha256"), head.pop("n_requests")
    if head.get("format") != TRACE_FORMAT:
        raise ValueError(f"unknown trace format {head.get('format')!r}")
    reqs = tuple(TraceRequest(d["id"], d["group"], tuple(d["tokens"])) for d in map(json.loads, lines[1:]))
    trace = Trace(head, reqs)
    if len(reqs) != n or trace.digest != want:
        raise ValueError(f"trace {path}: {len(reqs)} requests (header {n}), digest {trace.digest} (header {want})")
    return trace


# ---- offline statistics
def oracle_reuse(trace: Trace, chunk_size: int) -> list:
    """Per request, the restorable prefix length under an unbounded cache: the longest run of
    leading eligible chunks (first-token rule) that some earlier request stored. Every earlier
    request stores all of its complete chunks."""
    seen, out = set(), []
    for r in trace.requests:
        keys = chain_keys(STATS_NS_DIGEST, r.tokens, chunk_size)
        k = 0
        while k < max_restore_len(len(r.tokens), chunk_size) // chunk_size and keys[k] in seen:
            k += 1
        out.append(k * chunk_size)
        seen.update(keys)
    return out


def lru_restore_lengths(trace: Trace, chunk_size: int, capacity_chunks: int) -> list:
    """Per request, the restore length under the C1 policy (kvlab/gpu_prefix_cache.py) with room
    for `capacity_chunks` chunks: the longest present run of leading eligible chunks, touched in
    order; then each complete prompt chunk not present is inserted, evicting the least recently
    used. Present chunks are not touched by insertion. Key-level only (no tensors)."""
    lru, out = OrderedDict(), []
    for r in trace.requests:
        keys = chain_keys(STATS_NS_DIGEST, r.tokens, chunk_size)
        k = 0
        while k < max_restore_len(len(r.tokens), chunk_size) // chunk_size and keys[k] in lru:
            lru.move_to_end(keys[k])
            k += 1
        out.append(k * chunk_size)
        for key in keys:
            if key in lru or capacity_chunks <= 0:
                continue
            if len(lru) >= capacity_chunks:
                lru.popitem(last=False)
            lru[key] = True
    return out


def trace_stats(trace: Trace, chunk_size: int, *, chunk_bytes: int | None = None, budgets_bytes=()) -> dict:
    """Offline statistics, computed from the trace alone. With `chunk_bytes`, the working set in
    bytes and, per budget, the capacity in chunks, rho = working set / budget, and the C1 LRU
    restored tokens."""
    lens = sorted(len(r.tokens) for r in trace.requests)
    oracle = oracle_reuse(trace, chunk_size)
    chunks = set()
    for r in trace.requests:
        chunks.update(chain_keys(STATS_NS_DIGEST, r.tokens, chunk_size))
    total = sum(lens)
    out = {"trace_sha256": trace.digest, "generator": trace.header["generator"], "synthetic": True,
           "chunk_size": chunk_size, "n_requests": len(lens), "prompt_tokens": total,
           "prompt_len": {"min": lens[0], "median": statistics.median(lens), "max": lens[-1]} if lens else None,
           "oracle_reusable_tokens": sum(oracle), "oracle_reuse_ratio": sum(oracle) / total if total else 0.0,
           "requests_with_reuse": sum(1 for x in oracle if x), "unique_chunks": len(chunks)}
    if chunk_bytes is not None:
        ws = len(chunks) * chunk_bytes
        out["working_set_bytes"] = ws
        out["budgets"] = []
        for b in budgets_bytes:
            cap = b // chunk_bytes
            out["budgets"].append({"budget_bytes": b, "capacity_chunks": cap, "rho": ws / b if b else None,
                                   "lru_restored_tokens": sum(lru_restore_lengths(trace, chunk_size, cap))})
    return out
