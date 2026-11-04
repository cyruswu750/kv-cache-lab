import pytest

from kvlab.boundaries import eligible_boundaries, max_restore_len

C = 256


@pytest.mark.parametrize("n, expected", [
    (0, []), (1, []), (255, []),
    (256, []),            # exact multiple: the last (only) chunk must not be restored
    (257, [256]),
    (511, [256]),
    (512, [256]),         # exact multiple: chunk [256, 512) must not be restored
    (513, [256, 512]),
    (1024, [256, 512, 768]),
    (1025, [256, 512, 768, 1024]),
])
def test_eligible_boundaries(n, expected):
    assert eligible_boundaries(n, C) == expected
    assert max_restore_len(n, C) == (expected[-1] if expected else 0)


@pytest.mark.parametrize("n", range(0, 1100, 7))
def test_invariants(n):
    for L in eligible_boundaries(n, C):
        assert L % C == 0
        assert 0 < L <= n - 1  # at least one prompt token is always recomputed


def test_small_chunk():
    assert eligible_boundaries(5, 2) == [2, 4]
    assert eligible_boundaries(4, 2) == [2]


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_chunk_size(bad):
    with pytest.raises(ValueError):
        eligible_boundaries(10, bad)


def test_negative_prompt_len():
    with pytest.raises(ValueError):
        eligible_boundaries(-1, C)
