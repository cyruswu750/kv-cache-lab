import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kvlab.identity import ROOT_PARENT, canonical_json, chain_keys, chunk_key
from m1_toy import tokens, toy_ns

ROOT = Path(__file__).resolve().parent.parent


def test_digest_is_deterministic_across_processes_and_hash_seeds():
    code = ("import sys; sys.path[:0]=[sys.argv[1], sys.argv[1]+'/tests']\n"
            "from m1_toy import toy_ns, tokens\nfrom kvlab.identity import chain_keys\n"
            "ns=toy_ns(); print(ns.digest().hex(), *[k.hex() for k in chain_keys(ns.digest(), tokens(12), 4)])")
    outs = set()
    for seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, "-c", code, str(ROOT)], env=env, check=True,
                                capture_output=True, text=True).stdout)
    ns = toy_ns()
    local = " ".join([ns.digest().hex()] + [k.hex() for k in chain_keys(ns.digest(), tokens(12), 4)]) + "\n"
    assert outs == {local}


@pytest.mark.parametrize("field, value", [
    ("model_id", "other/model"), ("revision", "1" * 40), ("kv_dtype", "float16"), ("num_layers", 3),
    ("num_kv_heads", 2), ("head_dim", 16), ("chunk_size", 8), ("rope_parameters", {"rope_theta": 1e6, "rope_type": "default"}),
    ("layer_types", ("full_attention", "sliding_attention")), ("use_sliding_window", True), ("sliding_window", 4096),
    ("attn_implementation", "eager"), ("transformers_version", "5.22.2"), ("torch_version", "2.14.0"),
    ("layout_schema", "kvlab-kv-layout-v2"),
])
def test_every_namespace_field_changes_digest_and_keys(field, value):
    a = toy_ns()
    b = dataclasses.replace(a, **{field: value})
    assert a.digest() != b.digest()
    assert chain_keys(a.digest(), tokens(8), 4)[0] != chain_keys(b.digest(), tokens(8), 4)[0]


def test_same_chunk_tokens_after_different_prefixes_do_not_alias():
    d = toy_ns().digest()
    shared = [7, 7, 7, 7]
    k1 = chain_keys(d, [1, 2, 3, 4] + shared, 4)
    k2 = chain_keys(d, [1, 2, 3, 5] + shared, 4)
    assert k1[0] != k2[0] and k1[1] != k2[1]
    assert chunk_key(d, ROOT_PARENT, 0, shared) != k1[1]


def test_prefix_change_propagates_but_suffix_change_does_not_go_backwards():
    d = toy_ns().digest()
    base = tokens(12)
    changed_last = base[:8] + [999] + base[9:]
    a, b = chain_keys(d, base, 4), chain_keys(d, changed_last, 4)
    assert a[:2] == b[:2] and a[2] != b[2]
    assert chain_keys(d, base, 4) == a  # deterministic within a process


def test_only_complete_chunks_are_keyed():
    d = toy_ns().digest()
    assert len(chain_keys(d, tokens(7), 4)) == 1
    assert len(chain_keys(d, tokens(3), 4)) == 0
    assert chain_keys(d, tokens(9), 4) == chain_keys(d, tokens(8), 4)


def test_chunk_index_is_part_of_the_key():
    d = toy_ns().digest()
    assert chunk_key(d, ROOT_PARENT, 0, [1, 2]) != chunk_key(d, ROOT_PARENT, 1, [1, 2])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1, 2}, b"x", {1: "a"}, object()])
def test_canonical_json_rejects_non_canonical_values(bad):
    with pytest.raises((TypeError, ValueError)):
        canonical_json({"x": bad})


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2.5]}) == canonical_json({"a": [1, 2.5], "b": 1}) == b'{"a":[1,2.5],"b":1}'


@pytest.mark.parametrize("bad_token", [-1, 2**32, "5", 1.0])
def test_invalid_token_ids_rejected(bad_token):
    with pytest.raises(ValueError):
        chunk_key(toy_ns().digest(), ROOT_PARENT, 0, [1, bad_token])
