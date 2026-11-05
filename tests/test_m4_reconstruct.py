"""CPU tests for scripts/m4_reconstruct.py: rebuilding a frozen M4 tree from its manifest's base commit, and
comparing regenerated analysis outputs with archived ones. Fixture git repositories only; no torch, no GPU."""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("m4_reconstruct_under_test", ROOT / "scripts" / "m4_reconstruct.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

SOURCES = {"kvlab/m4_validate.py": "x = 1\n", "kvlab/other.py": "y = 2\n", "scripts/m4_analyze.py": "print(1)\n",
           "tests/test_a.py": "def test_a():\n    pass\n", "tests/gpu/test_b.py": "z = 3\n",
           "pyproject.toml": "[project]\nname = 'fixture'\n"}
DOCS = {"docs/P.md": "# protocol\n", "requirements.txt": "torch\n"}
ANALYSIS = ("kvlab/m4_validate.py", "scripts/m4_analyze.py")
RUN = {"batch_record.json": "{}\n", "session_e01.json": '{"a": 1}\n', "logs/session_e01.log": "ok\n"}
SPEC = "M4_EVAL_V2"


def _git(root, *args) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@invalid",
                           "-c", "commit.gpgsign=false", *args], cwd=root, env=env, check=True,
                          capture_output=True, text=True).stdout


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write(root: Path, files: dict) -> list:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return list(files)


def make_repo(dest: Path, *, edit=None, evidence_in_base=False, sidecar=b"") -> tuple:
    """A repository whose base commit is the frozen tree, with the manifest, sidecar and run directory written
    after it; a later commit then adds the evidence and changes a source file, a document and the source set
    (the post-archive state). Returns (root, manifest sha256)."""
    dest.mkdir(parents=True)
    _git(dest, "init", "-q")
    run = {f"results/m4/{SPEC}_run/{k}": v for k, v in RUN.items()}
    base_files = _write(dest, {**SOURCES, **DOCS, "README.md": "r\n", **(run if evidence_in_base else {})})
    _git(dest, "add", "--", *base_files)
    _git(dest, "commit", "-q", "-m", "base")
    m = {"experiment": SPEC, "base_git_head": _git(dest, "rev-parse", "HEAD").strip(),
         "git_diff_file": f"{SPEC}.freeze.diff", "git_diff_file_sha256": _sha(sidecar),
         "git_diff_head_sha256": _sha(sidecar), "git_status_porcelain": [], "untracked_files_sha256": {},
         "files_sha256": {k: _sha(v.encode()) for k, v in SOURCES.items()},
         "documents_sha256": {k: _sha(v.encode()) for k, v in DOCS.items()},
         "analysis_files_sha256": {k: _sha(SOURCES[k].encode()) for k in ANALYSIS},
         "run_dir": f"results/m4/{SPEC}_run"}
    if edit:
        edit(m)
    mbytes = json.dumps(m, indent=1, sort_keys=True).encode()
    (dest / "results/m4").mkdir(parents=True, exist_ok=True)
    (dest / f"results/m4/{SPEC}.freeze.json").write_bytes(mbytes)
    (dest / f"results/m4/{SPEC}.freeze.diff").write_bytes(sidecar)
    later = [f"results/m4/{SPEC}.freeze.json", f"results/m4/{SPEC}.freeze.diff"]
    later += _write(dest, {**({} if evidence_in_base else run), "kvlab/other.py": "y = 3\n",
                           "docs/P.md": "# changed after the freeze\n", "scripts/new_tool.py": "\n"})
    _git(dest, "add", "--", *later)
    _git(dest, "commit", "-q", "-m", "later: evidence and development")
    return dest, _sha(mbytes)


def build(repo, digest, out, spec=SPEC):
    return rc.main(["build", "--spec", spec, "--out", str(out), "--expect-freeze-sha256", digest], root=repo)


def test_build_rebuilds_the_base_revision_after_later_commits_and_copies_the_evidence(tmp_path, capsys):
    repo, digest = make_repo(tmp_path / "repo")
    out = tmp_path / "recon"
    assert build(repo, digest, out) == 0
    text = capsys.readouterr().out
    assert "VERIFIED" in text and "checked: 6 source files, 2 documents, 2 analysis files; copied 5" in text
    assert (out / "kvlab/other.py").read_text() == "y = 2\n"  # the frozen bytes, not the later commit's
    assert (out / "docs/P.md").read_text() == "# protocol\n"
    assert not (out / "scripts/new_tool.py").exists()
    for rel in [f"results/m4/{SPEC}.freeze.json", f"results/m4/{SPEC}.freeze.diff",
                *(f"results/m4/{SPEC}_run/{k}" for k in RUN)]:
        assert (out / rel).read_bytes() == (repo / rel).read_bytes()
    assert _git(repo, "status", "--porcelain") == ""  # the repository is not modified


def test_a_wrong_expected_digest_refuses_before_writing(tmp_path, capsys):
    repo, _ = make_repo(tmp_path / "repo")
    assert build(repo, "0" * 64, tmp_path / "recon") == 3
    assert "MISMATCH" in capsys.readouterr().out and not (tmp_path / "recon").exists()


def test_an_existing_output_directory_is_a_usage_error(tmp_path):
    repo, digest = make_repo(tmp_path / "repo")
    (tmp_path / "recon").mkdir()
    with pytest.raises(SystemExit) as e:
        build(repo, digest, tmp_path / "recon")
    assert e.value.code == 2 and list((tmp_path / "recon").iterdir()) == []


def test_an_unknown_spec_is_a_usage_error(tmp_path):
    repo, digest = make_repo(tmp_path / "repo")
    with pytest.raises(SystemExit) as e:
        build(repo, digest, tmp_path / "recon", spec="M4_EVAL_V3")
    assert e.value.code == 2


def _set(path, value):
    def edit(m):
        d = m
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return edit


def _drop(key, entry):
    return lambda m: m[key].pop(entry)


@pytest.mark.parametrize("edit, message, writes", [
    (_set(("files_sha256", "kvlab/other.py"), "0" * 64), "source kvlab/other.py", True),
    (_drop("files_sha256", "tests/gpu/test_b.py"), "source tests/gpu/test_b.py", True),  # extra file in the tree
    (_set(("files_sha256", "kvlab/gone.py"), "0" * 64), "source kvlab/gone.py", True),  # missing from the tree
    (_set(("documents_sha256", "docs/P.md"), "0" * 64), "documents_sha256 docs/P.md", True),
    (_set(("analysis_files_sha256", "kvlab/m4_validate.py"), "0" * 64), "analysis_files_sha256 kvlab/m4_va", True),
    (_set(("experiment",), "M4_EVAL_V1"), "manifest experiment", False),
    (_set(("untracked_files_sha256",), {"x.py": "0" * 64}), "untracked files", False),
    (_set(("git_status_porcelain",), [" M x.py"]), "dirty status", False),
    (_set(("base_git_head",), "f" * 40), "cannot be archived", False),
    (_set(("run_dir",), "results/m4/../../etc"), "not a relative path under results/m4", False),
    (_set(("run_dir",), "/abs/results/m4/x"), "not a relative path under results/m4", False),
    (_set(("analysis_files_sha256",), {}), "analysis_files_sha256 missing or empty", False),
])
def test_manifest_disagreements_are_mismatches(tmp_path, capsys, edit, message, writes):
    repo, digest = make_repo(tmp_path / "repo", edit=edit)
    assert build(repo, digest, tmp_path / "recon") == 3
    text = capsys.readouterr().out
    assert "MISMATCH" in text and message in text and "VERIFIED" not in text
    assert (tmp_path / "recon").exists() == writes


def test_a_nonempty_diff_sidecar_is_refused(tmp_path, capsys):
    repo, digest = make_repo(tmp_path / "repo", sidecar=b"diff --git a/x b/x\n")
    assert build(repo, digest, tmp_path / "recon") == 3
    assert "empty diff sidecar" in capsys.readouterr().out and not (tmp_path / "recon").exists()


def test_evidence_already_present_in_the_frozen_tree_is_a_mismatch(tmp_path, capsys):
    repo, digest = make_repo(tmp_path / "repo", evidence_in_base=True)
    assert build(repo, digest, tmp_path / "recon") == 3
    assert "already exists in the reconstructed tree" in capsys.readouterr().out


# ---- compare
A_ROOT, VALID = "/old/checkout", {"status": "valid", "frozen": True, "counts": {"sessions": 10}}


def outputs(d: Path, root: str, *, mean=-37.887153, verdict="A_faster", frozen=True, nan=float("nan"),
            md_extra=None) -> Path:
    d.mkdir(parents=True)
    (d / "m4_validation.json").write_text(json.dumps({**VALID, "frozen": frozen,
                                                      "run_dir": f"{root}/results/m4/{SPEC}_run"}, indent=1))
    (d / "m4_analysis.json").write_text(json.dumps({"primary": [{"mean": mean, "verdict": verdict, "sd": nan}]},
                                                   indent=1))
    (d / "m4_analysis.md").write_text("\n".join(["# analysis", f"| T1 | {mean} | {verdict} |",
                                                 *([md_extra] if md_extra else [])]) + "\n")
    return d


def compare(tmp_path, reg: Path, regenerated_root):
    return rc.main(["compare", "--archived", str(tmp_path / "a"), "--regenerated", str(reg),
                    "--archived-root", A_ROOT, "--regenerated-root", str(regenerated_root)])


def test_identical_outputs_compare_equal(tmp_path, capsys):
    outputs(tmp_path / "a", A_ROOT)
    assert compare(tmp_path, outputs(tmp_path / "b", A_ROOT), A_ROOT) == 0
    text = capsys.readouterr().out
    assert text.count("IDENTICAL") == 3 and "3 identical, 0 with path-only" in text


def test_the_recorded_run_directory_is_a_path_only_difference(tmp_path, capsys):
    new_root = tmp_path.resolve() / "recon"
    outputs(tmp_path / "a", A_ROOT)
    reg = outputs(tmp_path / "b", str(new_root))
    assert compare(tmp_path, reg, new_root) == 0
    text = capsys.readouterr().out
    assert f"PATH-ONLY m4_validation.json $.run_dir: '{A_ROOT}/results/m4/{SPEC}_run'" in text
    assert "2 identical, 1 with path-only differences, 0 with substantive" in text
    assert "RESULT: no substantive difference" in text


def test_validation_only_outputs_compare(tmp_path, capsys):
    """The V1 shape: an invalid batch has m4_validation.json only."""
    new_root = tmp_path.resolve() / "recon"
    a, reg = outputs(tmp_path / "a", A_ROOT), outputs(tmp_path / "b", str(new_root))
    for d in (a, reg):
        (d / "m4_analysis.json").unlink()
        (d / "m4_analysis.md").unlink()
    assert compare(tmp_path, reg, new_root) == 0
    assert "compared 1 files: 0 identical, 1 with path-only differences" in capsys.readouterr().out


def _relocate_elsewhere(where, new_root):
    def apply(a_dir, b_dir):
        if where == "markdown":
            for d, r in ((a_dir, A_ROOT), (b_dir, new_root)):
                with open(d / "m4_analysis.md", "a") as f:
                    f.write(f"run: {r}/results\n")
        else:  # another JSON field: nested in the analysis, or a second top-level field in the validation
            name, key = ("m4_analysis.json", "run_dir") if where == "analysis" else ("m4_validation.json", "other")
            for d, r in ((a_dir, A_ROOT), (b_dir, new_root)):
                v = json.loads((d / name).read_text())
                v[key] = f"{r}/results/m4/{SPEC}_run"
                (d / name).write_text(json.dumps(v, indent=1))
    return apply


@pytest.mark.parametrize("where, expected", [
    ("markdown", "DIFFERENT m4_analysis.md line 3"),
    ("analysis", "DIFFERENT m4_analysis.json $.run_dir"),
    ("validation", "DIFFERENT m4_validation.json $.other"),
])
def test_a_relocated_root_anywhere_else_is_substantive(tmp_path, capsys, where, expected):
    new_root = tmp_path.resolve() / "recon"
    a, reg = outputs(tmp_path / "a", A_ROOT), outputs(tmp_path / "b", str(new_root))
    _relocate_elsewhere(where, new_root)(a, reg)
    assert compare(tmp_path, reg, new_root) == 3
    assert expected in capsys.readouterr().out


def test_a_sibling_of_the_roots_is_not_a_relocation(tmp_path, capsys):
    """Review P2: '/old/checkout_backup/...' is not below '/old/checkout'."""
    new_root = tmp_path.resolve() / "recon"
    outputs(tmp_path / "a", A_ROOT + "_backup")
    reg = outputs(tmp_path / "b", str(new_root) + "_backup")
    assert compare(tmp_path, reg, new_root) == 3
    assert "DIFFERENT m4_validation.json $.run_dir" in capsys.readouterr().out


def test_a_formatting_change_beside_the_relocated_run_dir_is_substantive(tmp_path, capsys):
    new_root = tmp_path.resolve() / "recon"
    outputs(tmp_path / "a", A_ROOT)
    reg = outputs(tmp_path / "b", str(new_root))
    (reg / "m4_validation.json").write_text(json.dumps(json.loads((reg / "m4_validation.json").read_text()), indent=2))
    assert compare(tmp_path, reg, new_root) == 3
    text = capsys.readouterr().out
    assert "PATH-ONLY m4_validation.json $.run_dir" in text
    assert "DIFFERENT m4_validation.json (bytes differ beyond $.run_dir)" in text


@pytest.mark.parametrize("keep_a, keep_b", [((), ()), (("m4_analysis.md",), ("m4_analysis.md",)),
                                            (("m4_validation.json",), ())])
def test_directories_without_a_validation_output_are_not_compared(tmp_path, capsys, keep_a, keep_b):
    """Review P2: two empty (or validation-less) directories must not count as a reproduction."""
    a, reg = outputs(tmp_path / "a", A_ROOT), outputs(tmp_path / "b", A_ROOT)
    for d, keep in ((a, keep_a), (reg, keep_b)):
        for p in list(d.iterdir()):
            if p.name not in keep:
                p.unlink()
    assert compare(tmp_path, reg, A_ROOT) == 3
    text = capsys.readouterr().out
    assert "RESULT: NOT COMPARED" in text and "no substantive difference" not in text


@pytest.mark.parametrize("change, where", [
    ({"mean": -37.887152}, "m4_analysis.json $.primary[0].mean"),  # a last-digit numeric change
    ({"verdict": "inconclusive"}, "m4_analysis.json $.primary[0].verdict"),
    ({"frozen": 1}, "m4_validation.json $.frozen"),  # True vs 1: the type differs
    ({"nan": 0.0}, "m4_analysis.json $.primary[0].sd"),
    ({"md_extra": "extra line"}, "m4_analysis.md (line count)"),
])
def test_substantive_differences_fail(tmp_path, capsys, change, where):
    outputs(tmp_path / "a", A_ROOT)
    assert compare(tmp_path, outputs(tmp_path / "b", A_ROOT, **change), A_ROOT) == 3
    text = capsys.readouterr().out
    assert f"DIFFERENT {where}" in text and "RESULT: SUBSTANTIVE DIFFERENCE" in text


def test_a_markdown_only_line_change_is_substantive(tmp_path, capsys):
    outputs(tmp_path / "a", A_ROOT, md_extra=f"note: {A_ROOT} verdict A_faster")
    assert compare(tmp_path, outputs(tmp_path / "b", A_ROOT, md_extra=f"note: {A_ROOT} verdict A_slower"),
                   A_ROOT) == 3
    text = capsys.readouterr().out
    assert "DIFFERENT m4_analysis.md line 3" in text and "2 identical, 0 with path-only differences, 1 with substantive" in text


def test_a_path_change_plus_another_change_in_the_same_string_is_substantive(tmp_path, capsys):
    new_root = tmp_path.resolve() / "recon"
    a = outputs(tmp_path / "a", A_ROOT)
    reg = outputs(tmp_path / "b", str(new_root))
    v = json.loads((reg / "m4_validation.json").read_text())
    v["run_dir"] = f"{new_root}/results/m4/OTHER_run"
    (reg / "m4_validation.json").write_text(json.dumps(v, indent=1))
    assert a.is_dir() and compare(tmp_path, reg, new_root) == 3
    assert "DIFFERENT m4_validation.json $.run_dir" in capsys.readouterr().out


def test_a_missing_or_extra_file_is_substantive(tmp_path, capsys):
    outputs(tmp_path / "a", A_ROOT)
    reg = outputs(tmp_path / "b", A_ROOT)
    (reg / "m4_analysis.md").unlink()
    (reg / "extra.json").write_text("{}")
    assert compare(tmp_path, reg, A_ROOT) == 3
    text = capsys.readouterr().out
    assert "DIFFERENT m4_analysis.md (file): 'archived present' -> 'regenerated absent'" in text
    assert "DIFFERENT extra.json (file): 'archived absent' -> 'regenerated present'" in text


def test_reformatted_but_equal_json_is_still_reported(tmp_path, capsys):
    outputs(tmp_path / "a", A_ROOT)
    reg = outputs(tmp_path / "b", A_ROOT)
    (reg / "m4_validation.json").write_text(json.dumps(json.loads((reg / "m4_validation.json").read_text())))
    assert compare(tmp_path, reg, A_ROOT) == 3
    assert "(bytes differ, values equal)" in capsys.readouterr().out
