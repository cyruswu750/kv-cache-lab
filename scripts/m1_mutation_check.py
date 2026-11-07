"""Mutation check for M1 CPU tests: apply each deliberate bug to a scratch copy and
confirm at least one test fails (KILLED).
Usage: .venv/bin/python scripts/m1_mutation_check.py <empty scratch dir>
"""
import shutil, subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent  # the repository root (this file is scripts/m1_mutation_check.py)
WORK = Path(sys.argv[1])
MUTANTS = [
 ("header CRC not checked", "kvlab/chunk_format.py", "    if zlib.crc32(rest, zlib.crc32(head[:FIXED.size])) != header_crc:\n        raise ChunkReadError(\"header_crc\")", "    pass", "tests/test_chunk_format.py"),
 ("payload CRC not checked", "kvlab/chunk_format.py", "    if zlib.crc32(dest) != payload_crc:\n        raise ChunkReadError(\"payload_crc\")", "    pass", "tests/test_chunk_format.py"),
 ("tmp bytes not reserved", "kvlab/disk_store.py", "        self.tmp_bytes += size  # reserved", "        pass  # reserved", "tests/test_disk_store.py"),
 ("tmp not cleaned on failure", "kvlab/disk_store.py", "            if not committed and self._unlink(tmp.name, \"tmp\") is not None:", "            if False:", "tests/test_disk_store.py"),
 ("pins ignored by eviction", "kvlab/host_store.py", "if e.pins == 0), None)", "), None)", "tests/test_host_store.py"),
 ("lease not released on error", "kvlab/host_store.py", "        try:\n            yield self.slot_view(entry.slot), entry.ident\n        finally:\n            entry.pins -= 1", "        yield self.slot_view(entry.slot), entry.ident\n        entry.pins -= 1", "tests/test_host_store.py"),
 ("restore continues past invalid chunk", "kvlab/tiered_cache.py", "                    self.counters.miss_reasons[reason] += 1\n                    break", "                    self.counters.miss_reasons[reason] += 1\n                    continue", "tests/test_tiered_cpu.py"),
 ("parent not hashed into key", "kvlab/identity.py", "    h.update(parent_key)\n", "", "tests/test_identity.py"),
 # M1.1 (external review): deletion accounting, marker/symlink handling, host accounting
 ("M1.1 bytes released although the file remains", "kvlab/disk_store.py", "        if err is not None:\n            self.undeletable[name] = size\n", "", "tests/test_disk_store.py"),
 ("M1.1 temp reservation released although the temp file remains", "kvlab/disk_store.py", "                self.undeletable[tmp.name] = size", "                pass", "tests/test_disk_store.py"),
 ("M1.1 unlink OSError escapes the store", "kvlab/disk_store.py", "        except OSError as e:\n            err = _errname(e)", "        except ValueError as e:\n            err = _errname(e)", "tests/test_engine_cpu.py"),
 ("M1.1 undeletable victim kept readable and retried", "kvlab/disk_store.py", "            self.undeletable[name] = size\n", "            self.entries[key] = size\n            self.committed_bytes += size\n", "tests/test_disk_store.py"),
 ("M1.1 marker symlink followed", "kvlab/disk_store.py", "        if not marker.is_file(follow_symlinks=False):", "        if not marker.is_file():", "tests/test_disk_store.py"),
 ("M1.1 chunk read follows symlinks", "kvlab/disk_store.py", "os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC", "os.O_RDONLY | os.O_CLOEXEC", "tests/test_disk_store.py"),
 ("M1.1 host peak resident not tracked", "kvlab/host_store.py", "        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)", "        pass", "tests/test_host_store.py"),
 # M1.2 (BUGLOG VAL-004): reintroduce the old evidence-runner behaviors into scripts/m1_run.py
 ("M1.2 request count reported as records + 1", "scripts/m1_run.py", "        \"requests\": len(records),", "        \"requests\": len(records) + 1,", "tests/test_m1_run_validation.py"),
 ("M1.2 logits compared with zip() only", "scripts/m1_run.py", "               lengths_ok and all(torch.equal(a, b) for a, b in zip(step_logits, ref_steps)),", "               all(torch.equal(a, b) for a, b in zip(step_logits, ref_steps)),", "tests/test_m1_run_validation.py"),
 ("M1.2 forward calls not part of acceptance", "scripts/m1_run.py", "            if not (rec.get(\"forward_calls\") == rec.get(\"expected_forward_calls\") == STEPS):", "            if False:", "tests/test_m1_run_validation.py"),
 ("M1.2 acceptance = strict_pass of present records only", "scripts/m1_run.py", "        \"acceptance\": not problems,", "        \"acceptance\": all(rec.get(\"strict_pass\") is True for rec in records),", "tests/test_m1_run_validation.py"),
 ("M1.2 fault occurrence not checked", "scripts/m1_run.py", "            if got != reason:", "            if False:", "tests/test_m1_run_validation.py"),
 ("namespace field ignored (attn_implementation)", "kvlab/identity.py", "        d[\"layer_types\"] = list(self.layer_types)\n", "        d[\"layer_types\"] = list(self.layer_types)\n        d.pop(\"attn_implementation\")\n", "tests/test_identity.py"),
]
results = []
for name, rel, old, new, test in MUTANTS:
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    for d in ("kvlab", "tests", "scripts"): shutil.copytree(SRC / d, WORK / d, ignore=shutil.ignore_patterns("__pycache__", "gpu"))
    shutil.copy(SRC / "pyproject.toml", WORK)
    evidence = "results/m1_1/m1_run_20261103T112930.json"  # read by tests/test_m1_run_validation.py
    (WORK / evidence).parent.mkdir(parents=True); shutil.copy(SRC / evidence, WORK / evidence)
    p = WORK / rel; s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))
    r = subprocess.run([str(SRC / ".venv/bin/python"), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test],
                       cwd=WORK, capture_output=True, text=True, timeout=300)
    last = [l for l in r.stdout.splitlines() if l.strip()][-1]
    results.append((name, "KILLED" if r.returncode != 0 else "SURVIVED", last))
for r in results: print(f"{r[1]:9s} {r[0]:45s} {r[2]}")
