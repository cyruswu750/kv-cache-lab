"""M2.1: duration of slow regimes in diagnostic sessions (CPU only, read only).

Usage: .venv/bin/python scripts/m2_1_regimes.py results/m2_1/<diag run> [...] > <out>.txt
A regime is a maximal chronological run of restore-path (host, disk) samples with the same
slow label (slow = e2e > 1.3x the min of its path/mode/case over the run's sessions).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kvlab import diag_harness as dh  # noqa: E402

for run in map(Path, sys.argv[1:]):
    samples = []
    for f in sorted(run.glob("session_*.json")):
        d = json.loads(f.read_text())
        samples += [s for s in d["samples"] if s.get("stage") == "diag" and not s.get("warmup")
                    and s["path"] in ("host", "disk") and not dh.sample_problems(s)]
    mins = defaultdict(lambda: float("inf"))
    for s in samples:
        mins[(s["path"], s["mode"], s["L"])] = min(mins[(s["path"], s["mode"], s["L"])], s["e2e_ns"])
    out = {"run": str(run), "n_restore_samples": len(samples)}
    durs = {"slow": [], "fast": []}
    for sid in sorted({s["session"] for s in samples}):
        seq = sorted((s for s in samples if s["session"] == sid), key=lambda s: s["t_wall"])
        lab = [s["e2e_ns"] > 1.3 * mins[(s["path"], s["mode"], s["L"])] for s in seq]
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and lab[j + 1] == lab[i]:
                j += 1
            end = seq[j]["t_wall"] + seq[j]["e2e_ns"] / 1e9
            durs["slow" if lab[i] else "fast"].append((end - seq[i]["t_wall"], j - i + 1))
            i = j + 1
    for k, v in durs.items():
        s = np.array([x[0] for x in v])
        n = np.array([x[1] for x in v])
        out[k] = {"regimes": len(v), "duration_s_median": float(np.median(s)), "duration_s_p75": float(np.percentile(s, 75)),
                  "duration_s_max": float(s.max()), "samples_median": float(np.median(n)), "samples_max": int(n.max()),
                  "regimes_with_ge_5_samples": int((n >= 5).sum()),
                  "duration_s_median_of_ge_5_sample_regimes": float(np.median(s[n >= 5])) if (n >= 5).any() else None}
    print(json.dumps(out, indent=1))
