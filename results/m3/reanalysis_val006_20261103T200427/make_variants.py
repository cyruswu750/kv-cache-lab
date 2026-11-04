"""Build scratch copies of the M3 evaluation sessions for the VAL-006 reproductions (never touches the source)."""
import json, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
(dst / "empty").mkdir(parents=True)
(dst / "control").mkdir()
for p in sorted(src.glob("session_*.json")):
    shutil.copy2(p, dst / "control" / p.name)
(dst / "no1024").mkdir()
for i in range(1, 6):
    r = json.loads((src / f"session_s{i}.json").read_text())
    n0 = len(r["samples"])
    r["samples"] = [s for s in r["samples"] if not (s["L"] == 1024 and s["S"] == 1)]
    (dst / "no1024" / f"session_s{i}.json").write_text(json.dumps(r, indent=1, default=str))
    print(f"s{i}: removed {n0 - len(r['samples'])} samples with (L, S) = (1024, 1)")
