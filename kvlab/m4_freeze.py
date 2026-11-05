"""Premeasurement freeze manifest and launch check for M4-B (the D31 pattern; CPU only, no torch).

`freeze()` refuses while the protocol's first line contains "DRAFT". The manifest records the tree
to be measured
(base HEAD, `git diff HEAD --binary` and its sidecar, every source file, untracked files, the
protocol documents, the analysis files, the stack and the exact commands). `verify_tree()` must
pass before `run`; the analysis rejects the evidence while an analysis file or document differs
from its frozen bytes (`manifest_problems`).

Reused from kvlab.m3_confirm (imported, never edited): source identity, package versions, GPU
identity, hashing, the atomic writer and the Refused exception.
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from kvlab import m3_confirm as mc
from kvlab import m4_spec as ms

MANIFEST_VERSION = 1
DRAFT_MARKER = "DRAFT"
DOCUMENTS = ("docs/M4B_PROTOCOL.md", "requirements.txt")
ANALYSIS_FILES = ("kvlab/m4_validate.py", "kvlab/m4_stats.py", "kvlab/m4_analysis.py", "kvlab/m4_spec.py",
                  "kvlab/m4_freeze.py",
                  "kvlab/trace_models.py", "kvlab/traces.py", "kvlab/boundaries.py", "kvlab/identity.py",
                  "kvlab/m3_confirm.py", "kvlab/m3_stats.py", "kvlab/perf_stats.py", "scripts/m4_analyze.py")
MODEL_ID, REVISION = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def manifest_path(spec: dict) -> str:
    return str(Path(spec["run_dir"]).parent / f"{spec['name']}.freeze.json")


def sidecar_path(spec: dict) -> str:
    return str(Path(manifest_path(spec)).with_suffix(".diff"))


def weight_snapshot(hf_home) -> dict:
    snap = Path(hf_home) / "hub" / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION
    if not snap.is_dir():
        return {"error": f"no snapshot dir {snap}"}
    return {p.name: {"blob": os.path.basename(os.readlink(p)) if p.is_symlink() else None, "bytes": p.stat().st_size}
            for p in sorted(snap.iterdir())}


def current_stack(root: Path) -> dict:
    """The recorded stack. Queries nvidia-smi read-only (no CUDA initialisation); not used by tests."""
    hf = os.environ.get("HF_HOME", str(Path(root) / ".hf_cache"))
    return {"python": platform.python_version(), "packages": mc.package_versions(), "weights": weight_snapshot(hf),
            **mc.gpu_identity(), "kernel": platform.release(), "cpu": mc.cpu_model()}


def _untracked(root: Path) -> list:
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root, capture_output=True,
                         text=True).stdout.splitlines()
    return sorted(out)


def build_manifest(root: Path, spec: dict, *, stack: dict, now=None, require_frozen_protocol=True):
    root = Path(root)
    problems = []
    doc = root / spec["protocol_doc"]
    if require_frozen_protocol and (not doc.is_file() or DRAFT_MARKER in doc.read_text().splitlines()[0]):
        problems.append(f"{spec['protocol_doc']} is still marked {DRAFT_MARKER} (or missing): refusing to freeze")
    for rel in (manifest_path(spec), sidecar_path(spec), spec["run_dir"]):
        if (root / rel).exists():
            problems.append(f"{rel} already exists")
    if problems:
        raise mc.Refused(problems)
    src = mc.source_identity(root)
    diff = mc._git(root, "diff", "HEAD", "--binary")
    now = time.time() if now is None else now
    manifest = {
        "manifest_version": MANIFEST_VERSION, "experiment": spec["name"], "spec_sha256": spec["spec_sha256"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), "frozen_unix": now,
        "base_git_head": src["git_head"], "git_status_porcelain": src["git_status_porcelain"],
        "git_diff_head_sha256": src["git_diff_head_sha256"], "git_diff_file": Path(sidecar_path(spec)).name,
        "files_sha256": src["files_sha256"],
        "untracked_files_sha256": {f: mc.sha256_file(root / f) for f in _untracked(root) if (root / f).is_file()},
        "documents_sha256": {d: mc.sha256_file(root / d) for d in DOCUMENTS},
        "analysis_files_sha256": {f: mc.sha256_file(root / f) for f in ANALYSIS_FILES},
        "stack": stack, "run_dir": spec["run_dir"], "budget_s": spec["budget_s"], "commands": ms.commands_for(spec),
        "note": "hashes of the working tree at freeze time; no commit is required; the manifest does not hash itself"}
    return manifest, diff


def freeze(root: Path, spec: dict, *, stack: dict, now=None) -> Path:
    manifest, diff = build_manifest(root, spec, stack=stack, now=now)
    path = Path(root) / manifest_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    side = Path(root) / sidecar_path(spec)
    with open(side, "x") as f:
        f.write(diff)
    manifest["git_diff_file_sha256"] = mc.sha256_bytes(diff.encode())
    with open(path, "x") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def manifest_problems(manifest, spec: dict, root: Path, *, current_analysis=True) -> list:
    if not isinstance(manifest, dict):
        return ["freeze manifest is not a JSON object"]
    out = []
    for k, want in (("manifest_version", MANIFEST_VERSION), ("experiment", spec["name"]),
                    ("spec_sha256", spec["spec_sha256"]), ("run_dir", spec["run_dir"]),
                    ("budget_s", spec["budget_s"]), ("commands", ms.commands_for(spec))):
        if manifest.get(k) != want:
            out.append(f"freeze manifest {k} {manifest.get(k)!r} differs from the spec ({want!r})")
    if not mc.finite(manifest.get("frozen_unix")):
        out.append("freeze manifest frozen_unix missing or not finite")
    if current_analysis:
        for key, files in (("analysis_files_sha256", ANALYSIS_FILES), ("documents_sha256", DOCUMENTS)):
            rec = manifest.get(key) if isinstance(manifest.get(key), dict) else {}
            if set(rec) != set(files):
                out.append(f"freeze manifest {key} lists {sorted(rec)}, expected {sorted(files)}")
            for f in files:
                p = Path(root) / f
                if not p.is_file() or rec.get(f) != mc.sha256_file(p):
                    out.append(f"{f} differs from its frozen bytes")
    return out


def verify_tree(root: Path, manifest: dict, spec: dict, *, stack: dict) -> list:
    root = Path(root)
    out = manifest_problems(manifest, spec, root)
    src = mc.source_identity(root)
    for k in ("git_head", "git_diff_head_sha256", "files_sha256"):
        mk = "base_git_head" if k == "git_head" else k
        if src[k] != manifest.get(mk):
            out.append(f"{k} differs from the freeze")
    for f, h in (manifest.get("untracked_files_sha256") or {}).items():
        if not (root / f).is_file() or mc.sha256_file(root / f) != h:
            out.append(f"untracked file {f} differs from the freeze")
    side = root / sidecar_path(spec)
    if not side.is_file() or mc.sha256_file(side) != manifest.get("git_diff_file_sha256"):
        out.append("diff sidecar missing or differs")
    if stack != manifest.get("stack"):
        out.append("current stack differs from the frozen stack")
    return out
