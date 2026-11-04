"""VAL-006: compare the fixed analyzer's outputs with the historical M3 outputs (read-only).

Usage: .venv/bin/python <this file> <historical run dir> <new output dir>
- m3_analysis.json: every historical field must be present and exactly equal (floats compared
  exactly). The only allowed differences are the keys the fix adds:
  - top level: validation, out_dir;
  - decision: evidence_status, spec, missing_from_analysis;
  - sessions.*: correctness_all_recomputed.
- samples_chronological.csv: byte-identical.
- m3_analysis.md: identical except for lines the fix adds.
Exit 0 only if all three hold.
"""
import difflib
import json
import sys
from pathlib import Path

ADDED = {(): {"validation", "out_dir"}, ("decision",): {"evidence_status", "spec", "missing_from_analysis"}}
ADDED_SESSION = {"correctness_all_recomputed"}


def walk(old, new, path, bad, added):
    if isinstance(old, dict) and isinstance(new, dict):
        allowed = ADDED.get(path, set()) | (ADDED_SESSION if len(path) == 2 and path[0] == "sessions" else set())
        for k in old:
            if k not in new:
                bad.append(f"{'/'.join(path + (k,))}: missing in new output")
            else:
                walk(old[k], new[k], path + (k,), bad, added)
        for k in new:
            if k not in old:
                (added if k in allowed else bad).append("/".join(path + (k,)))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            walk(a, b, path + (str(i),), bad, added)
    elif type(old) is not type(new) or old != new:
        bad.append(f"{'/'.join(path)}: {old!r} -> {new!r}")


def main():
    h, n = Path(sys.argv[1]), Path(sys.argv[2])
    old, new = json.loads((h / "m3_analysis.json").read_text()), json.loads((n / "m3_analysis.json").read_text())
    bad, added = [], []
    walk(old, new, (), bad, added)
    leaves = sum(1 for _ in leaves_of(old))
    print(f"m3_analysis.json: {leaves} historical leaf values compared exactly; differences: {len(bad)}")
    for b in bad[:20]:
        print("  DIFF", b)
    print(f"  added keys ({len(added)}): {sorted(set(a if not a.startswith('sessions/') else 'sessions/*/' + a.split('/')[-1] for a in added))}")
    d_old, d_new = old["decision"], new["decision"]
    print(f"  decision historical: supports_adoption={d_old['supports_adoption']}, required_but_not_established={d_old['required_but_not_established']}")
    print(f"  decision new       : supports_adoption={d_new['supports_adoption']}, evidence_status={d_new['evidence_status']}, "
          f"missing_from_analysis={d_new['missing_from_analysis']}")
    csv_same = (h / "samples_chronological.csv").read_bytes() == (n / "samples_chronological.csv").read_bytes()
    print(f"samples_chronological.csv byte-identical: {csv_same}")
    a, b = (h / "m3_analysis.md").read_text().splitlines(), (n / "m3_analysis.md").read_text().splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    md_only_added = all(op[0] == "insert" for op in ops)
    print(f"m3_analysis.md: {len(a)} historical lines, {len(b)} new; historical lines all kept unchanged: {md_only_added}")
    for tag, i1, i2, j1, j2 in ops:
        for line in b[j1:j2]:
            print(f"  {tag.upper()} +{line}")
        for line in a[i1:i2]:
            print(f"  {tag.upper()} -{line}")
    ok = not bad and csv_same and md_only_added and d_new["supports_adoption"] is False and \
        d_new["required_but_not_established"] == d_old["required_but_not_established"]
    print("RESULT:", "PASS (numerical results and verdict unchanged)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def leaves_of(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from leaves_of(v)
    elif isinstance(x, list):
        for v in x:
            yield from leaves_of(v)
    else:
        yield x


if __name__ == "__main__":
    main()
