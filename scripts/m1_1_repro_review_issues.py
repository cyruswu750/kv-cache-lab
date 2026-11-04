"""Reproduce the two disk-tier issues from the M1 external review against a source tree.

Usage: .venv/bin/python scripts/m1_1_repro_review_issues.py <tree containing kvlab/>
  e.g. the current project (.) or a pre-fix tree extracted from backups/pre-m1_1-*.tar.gz.
Uses only temporary directories; the unlink failure is injected by replacing os.unlink
for one file name (no real permissions are changed).
"""
import errno
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from kvlab.disk_store import MARKER, DiskStore  # noqa: E402

print("issue 1: eviction unlink failure (8-byte budget)")
with tempfile.TemporaryDirectory() as td:
    d = DiskStore(Path(td) / "c", budget_bytes=8, max_entries=10)
    ka, kb = b"\xaa" * 32, b"\xbb" * 32
    print("  put a:", d.put(ka, b"", b"1234"), "committed_bytes", d.committed_bytes)
    victim = d.root / f"{ka.hex()}.kvc"
    real = os.unlink

    def deny(p, *a, **k):
        if Path(p).name == victim.name:
            raise OSError(errno.EACCES, "Permission denied", str(p))
        return real(p, *a, **k)
    os.unlink = deny
    try:
        print("  put b:", d.put(kb, b"", b"5678"))
    except Exception as e:
        print("  put b raised:", type(e).__name__, e)
    finally:
        os.unlink = real
    print("  after: committed_bytes", d.committed_bytes, "used_bytes", d.used_bytes,
          "| file a exists", victim.exists(), "size", victim.stat().st_size if victim.exists() else None)

print("issue 2: marker is a symlink to an external file")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    ext = td / "external.txt"
    ext.write_text("KEEP")
    root = td / "cache"
    root.mkdir()
    stale = root / ("b" * 64 + ".kvc")
    stale.write_bytes(b"stale")
    os.symlink(ext, root / MARKER)
    try:
        DiskStore(root, budget_bytes=100, max_entries=1)
        print("  constructed without error")
    except Exception as e:
        print("  construction raised:", type(e).__name__, e)
    print("  external file now:", repr(ext.read_text()[:40]), "| existing cache file still present:", stale.exists())
