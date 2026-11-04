"""Restore-boundary selection (pure Python, CPU-testable).

Rule (first-token rule, docs/design.md §2): at least the last prompt token is
always recomputed, and only complete chunks are restored. For prompt length N
and chunk size C, a restore length L is eligible iff

    L % C == 0  and  0 < L <= N - 1

L == 0 means "nothing to restore" (full recomputation).
"""


def eligible_boundaries(prompt_len: int, chunk_size: int) -> list[int]:
    """All eligible restore lengths, ascending. Empty if no chunk fits."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    max_len = max(prompt_len - 1, 0)
    return list(range(chunk_size, max_len + 1, chunk_size))


def max_restore_len(prompt_len: int, chunk_size: int) -> int:
    """Longest eligible restore length, or 0 if none."""
    bounds = eligible_boundaries(prompt_len, chunk_size)
    return bounds[-1] if bounds else 0
