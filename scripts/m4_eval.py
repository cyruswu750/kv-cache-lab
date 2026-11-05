#!/usr/bin/env python3
"""M4-B batch control: freeze / verify / run (the D31 pattern). NOT EXECUTED in M4-B preparation.

  .venv/bin/python scripts/m4_eval.py freeze --spec M4_EVAL_V1     # refuses while the protocol is DRAFT
  .venv/bin/python scripts/m4_eval.py verify --spec M4_EVAL_V1
  HF_HOME=<weights cache> .venv/bin/python scripts/m4_eval.py run --spec M4_EVAL_V1 --expect-freeze-sha256 <hex>

`run` executes the frozen session commands one after another with kvlab.m3_confirm.run_batch (reused
unchanged): one monotonic deadline (the spec's budget_s), each session in its own process group, no
retries, replacements or extensions; any nonzero exit, missing report, deadline or stop signal stops
the batch. Exit 0 only if every session succeeded; 3 otherwise; validity is judged by m4_analyze.py.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m4_freeze as mf  # noqa: E402
from kvlab import m4_spec as ms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("freeze", "verify", "run"))
    ap.add_argument("--spec", required=True)
    ap.add_argument("--expect-freeze-sha256")
    a = ap.parse_args()
    spec = ms.SPECS[a.spec]
    if a.command == "freeze":
        try:
            path = mf.freeze(ROOT, spec, stack=mf.current_stack(ROOT))
        except mc.Refused as e:
            print("REFUSED:\n  " + "\n  ".join(e.problems))
            return 3
        print(f"{path}: sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
        return 0
    mp = ROOT / mf.manifest_path(spec)
    if not mp.is_file():
        print(f"REFUSED: no freeze manifest {mp}")
        return 3
    data = mp.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    manifest = json.loads(data)
    problems = mf.verify_tree(ROOT, manifest, spec, stack=mf.current_stack(ROOT))
    if a.command == "verify" or problems:
        print("verification: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
        return 0 if not problems and a.command == "verify" else 3
    if a.expect_freeze_sha256 != sha:
        print(f"REFUSED: manifest sha256 {sha} != --expect-freeze-sha256 {a.expect_freeze_sha256}")
        return 3
    rec = mc.run_batch(ROOT, manifest, sha, verification={"problems": problems})
    print(json.dumps({k: rec[k] for k in ("execution_status", "stop", "not_launched", "elapsed_s")}, default=str))
    return mc.exit_code(rec)


if __name__ == "__main__":
    sys.exit(main())
