"""M3_CONFIRM_V1 execution procedure (docs/M3_CONFIRM_PROTOCOL.md §8-§9). CPU only in this process.

Usage (from the repository root):
  .venv/bin/python scripts/m3_confirm.py freeze
      Once, after implementation and review, before any measurement. Writes the freeze manifest
      results/m3_confirm/M3_CONFIRM_V1.freeze.json and its diff sidecar (never overwritten).
      Refuses while the protocol is marked DRAFT, if tracked evidence under results/ changed,
      if the model or software stack differs from the required one, or if a manifest or the run
      directory already exists.
  .venv/bin/python scripts/m3_confirm.py verify [--freeze PATH]
      Read-only: does the working tree, the documents and the stack still equal the freeze?
  .venv/bin/python scripts/m3_confirm.py run [--freeze PATH] [--expect-freeze-sha256 HEX]
      Verifies as above, then runs the five frozen session commands one after another
      (scripts/m3_measure.py, unchanged), before one deadline: 1,800 s of monotonic time from
      launching the first session process. No retries, replacements or extensions. Any nonzero
      session exit, an exit 0 without a session file, the deadline, a stop signal (SIGINT,
      SIGTERM, SIGHUP) or a wrapper error stops the batch: a running session's own process
      group is terminated and reaped, and nothing further is launched.

Exit codes:
- 0: verify passed; or run: execution succeeded (every session exited 0 before the deadline,
  each with a session file);
- 1: refused (run: nothing launched, no run directory written);
- 2: usage;
- 3: run: execution failed (see batch_record.json "stop").

Whatever `run` returns, the evidence is classified only by scripts/m3_analyze.py (exit 0 valid,
3 invalid, 4 consistently recorded correctness failure). With `| tee`, capture the wrapper's own
status with ${PIPESTATUS[0]} (docs/M3_CONFIRM_PROTOCOL.md §11). The procedure never starts a GPU
process except through `run`.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m3_validate as mv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    v = sub.add_parser("verify")
    v.add_argument("--freeze", default=None)
    r = sub.add_parser("run")
    r.add_argument("--freeze", default=None)
    r.add_argument("--expect-freeze-sha256", default=None)
    args = ap.parse_args(argv)
    spec = mv.M3_CONFIRM_V1
    if args.cmd == "freeze":
        return mc.main_freeze(spec, ROOT)
    if args.cmd == "verify":
        return mc.main_verify(spec, ROOT, args.freeze)
    return mc.main_run(spec, ROOT, args.freeze, args.expect_freeze_sha256)


if __name__ == "__main__":
    sys.exit(main())
