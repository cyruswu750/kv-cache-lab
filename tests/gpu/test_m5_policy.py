"""M5 GPU correctness gate (real pinned model, cuda, pinned host slab). NOT RUN as of M5b preparation.

Run only through scripts/m5_gate.py (one 900 s deadline, fail-fast, after the M1 and M3 GPU suites):
  HF_HOME=<weights cache> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m5_gate.py --out-dir local/m5_gate_<ts>
The same bodies run on the CPU with a tiny random model in tests/test_m5_gate_cpu.py.
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import sys
from pathlib import Path

import pytest

from kvlab.tiered_cache import RESTORE_BACKENDS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tests/: the shared gate bodies
import m5_gate_cases as g  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.m5]


@pytest.fixture(scope="module")
def real():
    import torch
    from kvlab.identity import namespace_for
    from kvlab import m5_spec as sp
    assert torch.cuda.is_available(), "the M5 gate needs the GPU"
    model, _tok = km.load_model("cuda")
    ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, sp.C)
    return model, ns


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_forced_zero_intermediate_and_maximal_lengths(real, tmp_path, backend):
    g.check_forced_lengths(*real, tmp_path / "d", backend, pin=True)


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_cost_aware_wiring_with_a_synthetic_artifact(real, tmp_path, backend):
    g.check_cost_aware(*real, tmp_path / "d", backend, pin=True)


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_truncation_inside_and_no_read_beyond_the_selected_prefix(real, tmp_path, backend):
    g.check_truncation_and_no_read_beyond(*real, tmp_path, backend, pin=True)


def test_harness_smoke_sessions_pass_the_session_validator(real, tmp_path):
    from kvlab import m5_spec as sp
    g.check_harness_smoke(*real, tmp_path, pin=True, model_meta=sp.MODEL, require_cuda=True)
