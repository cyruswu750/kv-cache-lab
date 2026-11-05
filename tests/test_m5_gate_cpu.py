"""The M5 GPU-gate bodies (tests/m5_gate_cases.py) on the CPU with a tiny random Qwen2 and an unpinned host slab.
This validates the gate's logic before any GPU run; it is not a substitute for the GPU gate."""

import pytest

import m5_gate_cases as g
import m5b_toy as t
from kvlab.tiered_cache import RESTORE_BACKENDS


@pytest.fixture(scope="module")
def tiny():
    model = t.tiny_model()
    return model, t.tiny_ns(model)


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_forced_zero_intermediate_and_maximal_lengths(tiny, tmp_path, backend):
    g.check_forced_lengths(*tiny, tmp_path / "d", backend, pin=False)


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_cost_aware_wiring_with_a_synthetic_artifact(tiny, tmp_path, backend):
    g.check_cost_aware(*tiny, tmp_path / "d", backend, pin=False)


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_truncation_inside_and_no_read_beyond_the_selected_prefix(tiny, tmp_path, backend):
    g.check_truncation_and_no_read_beyond(*tiny, tmp_path, backend, pin=False)


def test_harness_smoke_sessions_pass_the_session_validator(tiny, tmp_path):
    g.check_harness_smoke(*tiny, tmp_path, pin=False, vocab=t.TINY_VOCAB, model_meta=t.TINY_MODEL, require_cuda=False)


def test_the_gpu_gate_collects_the_same_bodies():
    import ast
    from pathlib import Path
    src = (Path(__file__).parent / "gpu" / "test_m5_policy.py").read_text()
    called = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
              and n.value.id == "g"}
    assert called == {"check_forced_lengths", "check_cost_aware", "check_truncation_and_no_read_beyond",
                      "check_harness_smoke"}
