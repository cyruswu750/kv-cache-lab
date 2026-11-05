#!/usr/bin/env python3
"""Validate M4-B evidence, then (only if valid with no correctness failure) analyse it. CPU only.

  .venv/bin/python scripts/m4_analyze.py --spec M4_EVAL_V1 --run-dir results/m4/M4_EVAL_V1_run \\
      --out-dir <new dir> --expect-freeze-sha256 <hex>        # frozen evaluation
  ... --allow-unfrozen                                         # draft/tests only: labelled UNFROZEN

Exit 0 valid; 3 invalid or incomplete evidence (no statistics); 4 valid evidence with a strict
correctness failure (no statistics); 2 usage. Always writes m4_validation.json; m4_analysis.json
and .md only on exit 0. Refuses to write into an existing directory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m4_analysis as ma  # noqa: E402
from kvlab import m4_freeze as mf  # noqa: E402
from kvlab import m4_spec as ms  # noqa: E402
from kvlab import m4_validate as mv  # noqa: E402


def main(argv=None, root: Path = ROOT, specs=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--expect-freeze-sha256")
    g.add_argument("--allow-unfrozen", action="store_true")
    a = ap.parse_args(argv)
    spec = (specs or ms.SPECS)[a.spec]
    out = Path(a.out_dir)
    if out.exists():
        ap.error(f"{out} exists; the analysis never overwrites")
    manifest, sha, pre = None, None, []
    if a.expect_freeze_sha256:
        mp = Path(root) / mf.manifest_path(spec)
        try:
            data = mp.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            manifest = mc.load_json_finite(data)
        except (OSError, ValueError) as e:
            pre.append(f"freeze manifest unreadable: {e!r}")
        if sha is not None and sha != a.expect_freeze_sha256:
            pre.append(f"freeze manifest sha256 {sha} != expected {a.expect_freeze_sha256}")
        if manifest is not None:
            pre += mf.manifest_problems(manifest, spec, root)
    v = mv.validate(Path(root) / a.run_dir if not Path(a.run_dir).is_absolute() else Path(a.run_dir), spec,
                    manifest=manifest, manifest_sha256=sha, allow_unfrozen=a.allow_unfrozen)
    if pre:
        v["problems"] = pre + v["problems"]
        v["status"], v["sessions"] = "invalid", None
    code = mv.exit_code(v)
    out.mkdir(parents=True)
    (out / "m4_validation.json").write_text(json.dumps({k: x for k, x in v.items() if k != "sessions"}, indent=1))
    if code == 0:
        res = ma.analyze(v, spec)
        (out / "m4_analysis.json").write_text(json.dumps(res, indent=1))
        (out / "m4_analysis.md").write_text(ma.markdown(res))
    print(json.dumps({"status": v["status"], "exit": code, "problems": len(v["problems"]),
                      "correctness_failures": len(v["correctness_failures"]), "frozen": v["frozen"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
