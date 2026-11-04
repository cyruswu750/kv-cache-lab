"""GPU integration tests (run with: .venv/bin/python -m pytest -m gpu -v).

Needs CUDA and the pinned model in .hf_cache/. If CUDA is unavailable, tests are
SKIPPED with reason "BLOCKED"; that is not a pass.
Each test runs run_case(), which builds fresh caches for every path; no cache
object is shared between tests or paths.
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import pytest
import torch

from kvlab.experiment import case_passed, run_case
from kvlab.fixtures import BOUNDARY_TEXT, NATURAL_PROMPTS, chat_ids, exact_length_ids
from kvlab.tolerances import ATOL, RTOL

pytestmark = [pytest.mark.gpu, pytest.mark.m0_strict]
C = 256


@pytest.fixture(scope="module")
def model_tok():
    if not torch.cuda.is_available():
        pytest.skip("BLOCKED: CUDA not available")
    return km.load_model()


def _check(rec):
    assert rec["full"]["first_logits_finite"] and rec["full"]["all_step_logits_finite"]
    for r in rec["restores"]:
        assert rec["export_check"]["all_bytes_equal"], rec["export_check"]["mismatches"][:5]
        assert r["restore_check"]["all_bytes_equal"], r["restore_check"]["mismatches"][:5]
        assert r["suffix_len"] == rec["prompt_len"] - r["restore_len"] >= 1
        fl = r["first_logits"]
        assert fl["ref_finite"] and fl["test_finite"] and r["all_step_logits_finite"]
        assert fl["allclose"], fl
        assert fl["argmax_equal"], fl
        assert r["greedy"]["identical"], (r["greedy"], rec["full"]["greedy_tokens"], r["greedy_tokens"])
        if "negative_control_zero_kv" in r:
            assert not r["negative_control_zero_kv"]["allclose"], "restored K/V had no effect on logits"
    if rec["restores"]:
        assert rec["snapshot_unchanged"]
    assert case_passed(rec)


@pytest.mark.parametrize("n, expected_bounds", [
    (255, []), (256, []), (257, [256]), (511, [256]), (512, [256]), (513, [256, 512]),
])
def test_boundary_case(model_tok, n, expected_bounds):
    model, tok = model_tok
    rec = run_case(model, f"boundary_{n}", exact_length_ids(tok, BOUNDARY_TEXT, n), C, ATOL, RTOL)
    assert rec["prompt_len"] == n
    assert rec["eligible_restore_lens"] == expected_bounds
    assert [r["restore_len"] for r in rec["restores"]] == expected_bounds
    _check(rec)


@pytest.mark.parametrize("name", sorted(NATURAL_PROMPTS))
def test_natural_prompt(model_tok, name):
    model, tok = model_tok
    rec = run_case(model, name, chat_ids(tok, NATURAL_PROMPTS[name]), C, ATOL, RTOL)
    _check(rec)


def test_restore_is_used(model_tok):
    """The restored K/V must actually drive the suffix computation."""
    model, tok = model_tok
    rec = run_case(model, "restore_used_513", exact_length_ids(tok, BOUNDARY_TEXT, 513), C, ATOL, RTOL)
    full_restore = [r for r in rec["restores"] if r["restore_len"] == 512][0]
    assert full_restore["suffix_len"] == 1  # only one token was fed to the model
    neg = full_restore["negative_control_zero_kv"]
    assert not neg["allclose"] and neg["max_abs_err"] > ATOL
    assert full_restore["first_logits"]["allclose"]
