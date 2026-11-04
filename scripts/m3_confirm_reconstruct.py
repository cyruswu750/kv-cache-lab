#!/usr/bin/env python3
"""Reconstruct the frozen M3_CONFIRM_V1 source tree in a new directory and verify it byte for
byte against the freeze manifest (CPU only; no torch, no GPU).

Why: the frozen analysis (scripts/m3_analyze.py --spec M3_CONFIRM_V1) refuses to run while any
analysis file or protocol document differs from its frozen bytes. After development changes
those files, the analysis is reproduced from a reconstruction of the frozen tree instead. This
script never edits the repository, the manifest or the evidence. It never weakens a check.

Sources (exactly one):
  --untracked-tar TAR   base commit (manifest base_git_head) + the committed diff sidecar
                        (results/m3_confirm/M3_CONFIRM_V1.freeze.diff) + the four files that were
                        untracked at freeze time, taken from TAR (the freeze-time snapshot);
  --untracked-rev REV   the same, but the four files are taken from git revision REV;
  --tree-rev REV        the whole source tree of git revision REV (a commit whose tree should
                        equal the frozen tree).

Then:
- every manifest hash is checked: files_sha256 (and no extra file in the hashed set),
  untracked_files_sha256 and documents_sha256;
- the evidence the analysis reads is copied into the reconstruction and compared byte for byte:
  the manifest, the diff sidecar and the run directory.

Exit 0 = verified. Exit 3 = any mismatch (the reconstruction is not usable). Exit 2 = usage.

Usage:
  .venv/bin/python scripts/m3_confirm_reconstruct.py --out <new dir> \\
      --expect-freeze-sha256 <hex> (--untracked-tar <tar> | --untracked-rev <rev> | --tree-rev <rev>)
Then, from inside <new dir>, run that tree's own analyzer into another new directory:
  cd <new dir> && <repo>/.venv/bin/python scripts/m3_analyze.py --spec M3_CONFIRM_V1 \\
      --run-dir results/m3_confirm/M3_CONFIRM_V1_run --out-dir <new out dir> \\
      --expect-freeze-sha256 <hex>
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "results/m3_confirm/M3_CONFIRM_V1.freeze.json"
RUN_DIR = "results/m3_confirm/M3_CONFIRM_V1_run"
SOURCE_GLOBS = ("kvlab/*.py", "scripts/*.py", "tests/**/*.py")  # + pyproject.toml, as kvlab.m3_confirm.source_identity


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*args, cwd=ROOT, data=None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, input=data, capture_output=True, check=True).stdout


def extract_rev(rev: str, out: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", rev))) as t:
        t.extractall(out, filter="data")


def hashed_set(root: Path) -> dict:
    files = sorted({*[p for g in SOURCE_GLOBS for p in root.glob(g)], root / "pyproject.toml"})
    return {str(p.relative_to(root)): sha256(p.read_bytes()) for p in files if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect-freeze-sha256", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--untracked-tar", type=Path)
    src.add_argument("--untracked-rev")
    src.add_argument("--tree-rev")
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        ap.error(f"{out} exists; the reconstruction always goes into a new directory")

    problems = []
    mbytes = (ROOT / MANIFEST).read_bytes()
    if sha256(mbytes) != a.expect_freeze_sha256:
        print(f"MISMATCH: manifest sha256 {sha256(mbytes)} != expected {a.expect_freeze_sha256}")
        return 3
    m = json.loads(mbytes)
    side = ROOT / "results/m3_confirm" / m["git_diff_file"]
    side_bytes = side.read_bytes()
    if not (sha256(side_bytes) == m["git_diff_file_sha256"] == m["git_diff_head_sha256"]):
        print(f"MISMATCH: diff sidecar sha256 {sha256(side_bytes)} differs from the manifest")
        return 3

    out.mkdir(parents=True)
    if a.tree_rev:
        extract_rev(a.tree_rev, out)
        mode = f"tree of {a.tree_rev} ({git('rev-parse', a.tree_rev).decode().strip()})"
    else:
        extract_rev(m["base_git_head"], out)
        # Apply the sidecar inside a throwaway repository rooted at `out`, so that git cannot
        # resolve paths against an enclosing repository.
        git("init", "-q", cwd=out)
        git("apply", "--binary", "-", cwd=out, data=side_bytes)
        shutil.rmtree(out / ".git")
        names = sorted(m["untracked_files_sha256"])
        if a.untracked_tar:
            with tarfile.open(a.untracked_tar) as t:
                members = {x.name: x for x in t.getmembers() if x.isfile()}
                if sorted(members) != names:
                    problems.append(f"untracked tar holds {sorted(members)}, manifest lists {names}")
                for n in names:
                    if n in members:
                        (out / n).parent.mkdir(parents=True, exist_ok=True)
                        (out / n).write_bytes(t.extractfile(members[n]).read())
            mode = f"base {m['base_git_head']} + sidecar + untracked files from {a.untracked_tar}"
        else:
            for n in names:
                (out / n).parent.mkdir(parents=True, exist_ok=True)
                (out / n).write_bytes(git("show", f"{a.untracked_rev}:{n}"))
            mode = f"base {m['base_git_head']} + sidecar + untracked files from {a.untracked_rev}"

    got = hashed_set(out)
    want = m["files_sha256"]
    for f in sorted(set(got) | set(want)):
        if got.get(f) != want.get(f):
            problems.append(f"source {f}: reconstructed {got.get(f)} vs manifest {want.get(f)}")
    for key in ("untracked_files_sha256", "documents_sha256"):
        for f, h in m[key].items():
            p = out / f
            have = sha256(p.read_bytes()) if p.is_file() else None
            if have != h:
                problems.append(f"{key} {f}: reconstructed {have} vs manifest {h}")

    # evidence read by the analysis: copied, then compared byte for byte with the originals
    copied = 0
    for rel in [MANIFEST, f"results/m3_confirm/{m['git_diff_file']}",
                *sorted(str(p.relative_to(ROOT)) for p in (ROOT / RUN_DIR).rglob("*") if p.is_file())]:
        dst = out / rel
        if dst.exists():
            problems.append(f"{rel} already exists in the reconstructed tree")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
        if sha256(dst.read_bytes()) != sha256((ROOT / rel).read_bytes()):
            problems.append(f"copy of {rel} differs")
        copied += 1

    print(f"reconstruction: {mode}")
    print(f"out: {out}")
    print(f"manifest sha256: {sha256(mbytes)}")
    print(f"checked: {len(want)} source files, {len(m['untracked_files_sha256'])} untracked files, "
          f"{len(m['documents_sha256'])} documents; copied {copied} evidence files")
    if problems:
        print(f"MISMATCH ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 3
    print("VERIFIED: the reconstructed tree equals the frozen bytes recorded in the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
