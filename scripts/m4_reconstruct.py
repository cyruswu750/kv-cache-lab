#!/usr/bin/env python3
"""Rebuild the frozen source tree of an executed M4 evaluation and check it against its freeze manifest, or
compare regenerated analysis outputs with the archived ones (CPU only; no torch, no GPU, no model).

Why: `scripts/m4_eval.py verify` binds HEAD and the whole source set, so it fails once anything is committed after
the freeze (the evidence included). That is expected, and it is not weakened. The frozen validation and analysis
are reproduced instead from the frozen revision, by that revision's own analyzer. This script never edits the
repository, the manifest or the evidence, and it is not a live environment check: it does not query the GPU,
the driver or the installed packages.

build    extract `git archive <manifest base_git_head>` into a new directory. Both M4 freezes were made from a
         clean HEAD (an empty diff sidecar, no untracked file); any other manifest is refused. Then:
         - every manifest hash is checked: files_sha256 (and no extra file in the hashed set), documents_sha256
           and analysis_files_sha256;
         - the evidence the analysis reads (the manifest, its sidecar and the run directory) is copied in and
           compared byte for byte; it must not already exist in the frozen tree.
compare  compare an archived analysis output directory with a regenerated one, file by file: JSON value by
         value, other files line by line. Both directories must hold m4_validation.json. The only path-only
         difference is the top-level `run_dir` of m4_validation.json (the validator records the absolute run
         directory it read): both values must lie below --archived-root and --regenerated-root respectively,
         with the same relative part, and the file's bytes must otherwise be identical. Every other difference,
         Markdown included, and a missing or extra file, is substantive.

Exit 0 = verified / no substantive difference; 3 = a mismatch or a substantive difference; 2 = usage.

Usage (between the two steps, the reconstructed tree's own analyzer runs from inside the new directory):
  .venv/bin/python scripts/m4_reconstruct.py build --spec M4_EVAL_V2 --out <new dir> --expect-freeze-sha256 <hex>
  cd <new dir> && <repo>/.venv/bin/python scripts/m4_analyze.py --spec M4_EVAL_V2 \\
      --run-dir results/m4/M4_EVAL_V2_run --out-dir <new out dir> --expect-freeze-sha256 <hex>
  .venv/bin/python scripts/m4_reconstruct.py compare --archived results/m4/M4_EVAL_V2_analysis \\
      --regenerated <new out dir> --archived-root <root recorded in the archive> --regenerated-root <new dir>
"""

import argparse
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SPECS = ("M4_EVAL_V1", "M4_EVAL_V2")
SOURCE_GLOBS = ("kvlab/*.py", "scripts/*.py", "tests/**/*.py")  # + pyproject.toml, as kvlab.m3_confirm.source_identity
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
# The one recorded absolute path in the analysis outputs: the validator writes the run directory it read.
RELOCATABLE = ("m4_validation.json", "run_dir")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hashed_set(root: Path) -> dict:
    files = sorted({*[p for g in SOURCE_GLOBS for p in root.glob(g)], root / "pyproject.toml"})
    return {str(p.relative_to(root)): sha256(p.read_bytes()) for p in files if p.is_file()}


def build(spec: str, out: Path, expect: str, root: Path) -> int:
    rel_manifest = f"results/m4/{spec}.freeze.json"
    mbytes = (root / rel_manifest).read_bytes()
    if sha256(mbytes) != expect:
        print(f"MISMATCH: {rel_manifest} sha256 {sha256(mbytes)} != expected {expect}")
        return 3
    m = json.loads(mbytes)
    if not isinstance(m, dict):
        print(f"MISMATCH: {rel_manifest} is not a JSON object")
        return 3
    problems = [f"manifest {key} missing or empty" for key in ("files_sha256", "documents_sha256",
                                                               "analysis_files_sha256")
                if not (isinstance(m.get(key), dict) and m[key])]
    if m.get("experiment") != spec:
        problems.append(f"manifest experiment {m.get('experiment')!r} is not {spec!r}")
    run_dir = PurePosixPath(str(m.get("run_dir", "")))
    if run_dir.is_absolute() or ".." in run_dir.parts or run_dir.parts[:2] != ("results", "m4"):
        problems.append(f"manifest run_dir {str(run_dir)!r} is not a relative path under results/m4")
    side_rel = f"results/m4/{m.get('git_diff_file')}"
    side = (root / side_rel).read_bytes() if (root / side_rel).is_file() else None
    if side != b"" or not (m.get("git_diff_file_sha256") == m.get("git_diff_head_sha256") == EMPTY_SHA256):
        problems.append(f"{side_rel}: not the empty diff sidecar of a clean-HEAD freeze (unsupported here)")
    if m.get("untracked_files_sha256") or m.get("git_status_porcelain"):
        problems.append("the manifest records untracked files or a dirty status (unsupported here)")
    try:
        tar = subprocess.run(["git", "archive", "--format=tar", str(m.get("base_git_head"))], cwd=root,
                             capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        tar = None
        problems.append(f"base commit {m.get('base_git_head')!r} cannot be archived: {e.stderr.decode().strip()}")
    if problems:
        print(f"MISMATCH ({len(problems)}), nothing written:")
        for p in problems:
            print(f"  {p}")
        return 3

    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(out, filter="data")
    got, want = hashed_set(out), m["files_sha256"]
    for f in sorted(set(got) | set(want)):
        if got.get(f) != want.get(f):
            problems.append(f"source {f}: reconstructed {got.get(f)} vs manifest {want.get(f)}")
    for key in ("documents_sha256", "analysis_files_sha256"):
        for f, h in m[key].items():
            p = out / f
            have = sha256(p.read_bytes()) if p.is_file() else None
            if have != h:
                problems.append(f"{key} {f}: reconstructed {have} vs manifest {h}")

    evidence = sorted(str(p.relative_to(root)) for p in (root / run_dir).rglob("*") if p.is_file())
    if not evidence:
        problems.append(f"{run_dir}: no evidence file")
    copied = 0
    for rel in [rel_manifest, side_rel, *evidence]:
        dst = out / rel
        if dst.exists():
            problems.append(f"{rel} already exists in the reconstructed tree")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dst)
        if dst.read_bytes() != (root / rel).read_bytes():
            problems.append(f"copy of {rel} differs")
        copied += 1

    print(f"reconstruction: tree of {m['base_git_head']} (manifest base_git_head)")
    print(f"out: {out}")
    print(f"manifest sha256: {sha256(mbytes)}")
    print(f"checked: {len(want)} source files, {len(m['documents_sha256'])} documents, "
          f"{len(m['analysis_files_sha256'])} analysis files; copied {copied} evidence files "
          f"(manifest, sidecar, {len(evidence)} in {run_dir})")
    if problems:
        print(f"MISMATCH ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 3
    print("VERIFIED: the reconstructed tree equals the frozen bytes recorded in the manifest")
    return 0


def _short(x) -> str:
    r = repr(x)
    return r if len(r) <= 160 else r[:157] + "..."


def _diff_values(a, b, ptr: str, out: list) -> None:
    if type(a) is not type(b):
        out.append(("DIFFERENT", ptr, a, b))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in a and k in b:
                _diff_values(a[k], b[k], f"{ptr}.{k}", out)
            else:
                out.append(("DIFFERENT", f"{ptr}.{k}", a.get(k, "<absent>"), b.get(k, "<absent>")))
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(("DIFFERENT", f"{ptr} (length)", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_values(x, y, f"{ptr}[{i}]", out)
    elif isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return
    elif a != b:
        out.append(("DIFFERENT", ptr, a, b))


def _relocated(a, b, roots: tuple) -> bool:
    """a and b name the same location below the archived and the regenerated root respectively."""
    ra, rb = roots
    return (isinstance(a, str) and isinstance(b, str) and a.startswith(ra + "/") and b.startswith(rb + "/")
            and a[len(ra):] == b[len(rb):])


def _diff_file(name: str, ba: bytes, bb: bytes, roots: tuple) -> list:
    """Every difference between two differing files. The only path-only difference allowed is RELOCATABLE: the
    top-level `run_dir` of m4_validation.json, relocated from the archived to the regenerated root, with the
    file's bytes otherwise identical. Everything else, Markdown included, is substantive."""
    found = []
    if not name.endswith(".json"):
        la, lb = ba.decode().splitlines(), bb.decode().splitlines()
        if len(la) != len(lb):
            found.append(("DIFFERENT", "(line count)", len(la), len(lb)))
        found += [("DIFFERENT", f"line {i}", x, y) for i, (x, y) in enumerate(zip(la, lb), 1) if x != y]
        return found or [("DIFFERENT", "(bytes differ, lines equal)", "", "")]
    try:
        va, vb = json.loads(ba), json.loads(bb)
    except ValueError as e:
        return [("DIFFERENT", "(parse)", repr(e), "")]
    file, key = RELOCATABLE
    if (name == file and isinstance(va, dict) and isinstance(vb, dict) and key in va and key in vb
            and va[key] != vb[key] and _relocated(va[key], vb[key], roots)):
        found.append(("PATH-ONLY", f"$.{key}", va[key], vb[key]))
        if ba.replace(json.dumps(va[key]).encode(), json.dumps(vb[key]).encode()) != bb:
            found.append(("DIFFERENT", f"(bytes differ beyond $.{key})", "", ""))
        va = {**va, key: vb[key]}
    _diff_values(va, vb, "$", found)
    if not found:  # bytes differ but the parsed values are equal (formatting only)
        found.append(("DIFFERENT", "(bytes differ, values equal)", "", ""))
    return found


def compare(archived: Path, regenerated: Path, archived_root: str, regenerated_root: str) -> int:
    roots = (archived_root.rstrip("/"), str(Path(regenerated_root).resolve()))
    names_a = {str(p.relative_to(archived)) for p in archived.rglob("*") if p.is_file()}
    names_b = {str(p.relative_to(regenerated)) for p in regenerated.rglob("*") if p.is_file()}
    required = RELOCATABLE[0]
    if required not in names_a or required not in names_b:
        print(f"RESULT: NOT COMPARED: {required} must exist in both directories "
              f"(archived: {len(names_a)} files, regenerated: {len(names_b)} files)")
        return 3
    diffs, counts = [], {"identical": 0, "path-only": 0, "different": 0}
    for name in sorted(names_a | names_b):
        if name not in names_a or name not in names_b:
            diffs.append(("DIFFERENT", name, "(file)", "archived " + ("present" if name in names_a else "absent"),
                          "regenerated " + ("present" if name in names_b else "absent")))
            counts["different"] += 1
            continue
        ba, bb = (archived / name).read_bytes(), (regenerated / name).read_bytes()
        if ba == bb:
            print(f"IDENTICAL {name} (sha256 {sha256(ba)})")
            counts["identical"] += 1
            continue
        found = _diff_file(name, ba, bb, roots)
        counts["different" if any(f[0] == "DIFFERENT" for f in found) else "path-only"] += 1
        diffs += [(k, name, p, x, y) for k, p, x, y in found]
    for kind, name, ptr, x, y in diffs:
        print(f"{kind} {name} {ptr}: {_short(x)} -> {_short(y)}")
    print(f"compared {len(names_a | names_b)} files: {counts['identical']} identical, "
          f"{counts['path-only']} with path-only differences, {counts['different']} with substantive differences")
    if counts["different"]:
        print("RESULT: SUBSTANTIVE DIFFERENCE")
        return 3
    print("RESULT: no substantive difference" + (" (path-only differences listed above)" if counts["path-only"] else ""))
    return 0


def main(argv=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild and verify the frozen tree, then copy the evidence in")
    b.add_argument("--spec", required=True, choices=SPECS)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--expect-freeze-sha256", required=True)
    c = sub.add_parser("compare", help="compare archived and regenerated analysis outputs")
    c.add_argument("--archived", required=True, type=Path)
    c.add_argument("--regenerated", required=True, type=Path)
    c.add_argument("--archived-root", required=True)
    c.add_argument("--regenerated-root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        out = a.out.resolve()
        if out.exists():
            ap.error(f"{out} exists; the reconstruction always goes into a new directory")
        return build(a.spec, out, a.expect_freeze_sha256, Path(root))
    for d in (a.archived, a.regenerated):
        if not d.is_dir():
            ap.error(f"{d} is not a directory")
    if not a.archived_root.rstrip("/"):
        ap.error("--archived-root must not be empty")
    return compare(a.archived, a.regenerated, a.archived_root, a.regenerated_root)


if __name__ == "__main__":
    sys.exit(main())
