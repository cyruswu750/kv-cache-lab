"""M5b: benchmark-only state preparation, verification and reset for one case (docs/M5_PROTOCOL_DRAFT.md §4).

Nothing here changes a production store's behaviour. It uses only existing public operations:
`HostStore.allocate/insert/lease/discard`, `DiskStore.put` and `TieredKVCache` reads.

**Provenance (`ChunkBank`).** The source chunk bytes of a case come from ONE full prefill of `prompt[:P]`
(P covers every indexed chunk), copied by the unchanged M1 store path into an unpinned source host store. The
same bytes:
- fill the measured host slots and disk files;
- seed the `ResidentReference` (`reference()`), so restored bytes and outputs are checked against the source
  that was actually stored, never against an independent recomputation of another length.

**Preparation (`CaseState`).**
- Fresh stores with the declared capacities: host slots = indexed host chunks + expected admissions, and one
  staging buffer; disk files = indexed disk chunks.
- Host chunks are inserted in the declared LRU order (slots 0, 1, ...), and disk files are written in the
  declared LRU order.

**Reset after every sample (`reset`):**
- discard the sample's admitted chunks in reverse allocation order, which restores the free-slot list exactly;
- re-establish host recency by leasing each indexed host chunk in the declared order (a lease touches LRU and
  releases its pin);
- re-establish disk recency by `put` of each existing disk key in the declared order (the store's "exists"
  path only moves it to the end).

Admission cleanup alone would not restore LRU order. Reopening a `DiskStore` would delete the files, so it is
never used as a reset. Production counters are never reset or overwritten: callers take deltas.

**Verification:**
- **Structure** (`observe` / `check`, before every sample):
  - tier membership and LRU orders, the host slot mapping, the free list and capacities;
  - no pins, no busy staging buffer, no temp or undeletable bytes;
  - the directory listing, and every file's size, inode and mtime.

  This is compared with the setup observation and, for the spec-derivable part, with
  `m5_spec.expected_structure`. It proves the index/LRU/slot state is identical before every sample and that
  no file was replaced or rewritten. It does not read payloads.
- **Content** (`content_check`, after setup and at the end of the case): the SHA-256 of every indexed host
  slot and of every disk file's bytes, compared with the bank. Every sample also compares the bytes it
  actually restored with the bank (the strict check in kvlab/m5_bench.py).
"""

import hashlib
import io
import os
import shutil
from pathlib import Path

import torch

from kvlab import chunk_format as cf
from kvlab import m5_spec as sp
from kvlab.disk_store import MARKER, DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys
from kvlab.model import prefill
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import TieredKVCache


class StateError(RuntimeError):
    """The measured state differs from the declared one: the session must stop (evidence invalid)."""


def _sha(b) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


class ChunkBank:
    """Source bytes of every indexed chunk of one case, from one full prefill of prompt[:P]."""

    def __init__(self, model, ns, tokens, P: int):
        C = ns.chunk_size
        self.ns, self.C, self.P = ns, C, P
        self.digest = ns.digest()
        self.keys = chain_keys(self.digest, tokens, C)        # every complete chunk of the prompt
        n = P // C
        self.cb = cf.payload_len(ns)
        self.data = torch.empty((n, self.cb), dtype=torch.uint8)
        self.idents = [ChunkIdentity.for_chunk(self.digest, tokens, C, i, self.keys) for i in range(n)]
        if n:
            src_host = HostStore((n + 1) * self.cb, self.cb, n, pin=False)
            src = TieredKVCache(ns, src_host, None)
            device = next(model.parameters()).device
            _, cache = prefill(model, torch.tensor([list(tokens[:P])], device=device), None, 0)
            stored = src.store(cache, list(tokens[:P]), self.keys[:n])  # the M1 store path (D2H into slots)
            del cache
            if stored != list(range(n)):
                raise StateError(f"source store kept {stored}, expected all {n} chunks")
            for i in range(n):
                with src_host.lease(self.keys[i]) as (flat, ident):
                    if ident != self.idents[i]:
                        raise StateError("source identity mismatch")
                    self.data[i].copy_(flat)
            src_host.close()
        self.payload_sha256 = [_sha(self.data[i].numpy()) for i in range(n)]
        self.headers = [cf.encode_header(ns, self.idents[i]) for i in range(n)]
        self.file_sha256 = []
        for i in range(n):
            buf = io.BytesIO()
            cf.write_chunk(buf, self.headers[i], self.data[i].numpy())
            self.file_sha256.append(_sha(buf.getvalue()))

    def record(self) -> dict:
        return {"P": self.P, "payload_sha256": self.payload_sha256, "file_sha256": self.file_sha256,
                "method": "one full prefill of prompt[:P]; chunk bytes copied by the M1 store path"}

    def reference(self, model) -> ResidentReference:
        """A ResidentReference whose resident chunks ARE the bank's bytes (same provenance)."""
        ns = self.ns
        ref = ResidentReference(model, self.digest, self.C)
        device = next(model.parameters()).device
        dtype = getattr(torch, ns.kv_dtype)
        for i in range(self.data.shape[0]):
            t = self.data[i].view(dtype).view(ns.num_layers, 2, 1, ns.num_kv_heads, self.C, ns.head_dim)
            ref.chunks[self.keys[i]] = [(t[li, 0].to(device).clone(), t[li, 1].to(device).clone())
                                        for li in range(ns.num_layers)]
        return ref


class CaseState:
    """The measured stores of one case, prepared from a bank exactly as the spec declares."""

    def __init__(self, ns, case: dict, spec: dict, bank: ChunkBank, root: Path, *, pin: bool):
        self.ns, self.case, self.bank = ns, case, bank
        self.keys = bank.keys
        cb, self.file_bytes = cf.payload_len(ns), cf.file_size(ns)
        slots, files = case["capacities"]["host_slots"], case["capacities"]["disk_files"]
        self.root = Path(root)
        if self.root.exists():
            raise StateError(f"{self.root} exists; a case always starts from a new directory")
        self.host = HostStore((slots + 1) * cb, cb, max_entries=slots, pin=pin)
        self.disk = DiskStore(self.root, files * self.file_bytes, max_entries=files)
        group = spec["restore_staging_chunks"] if spec["restore_staging_chunks"] is not None else 4
        self.tiered = TieredKVCache(ns, self.host, self.disk, restore_backend=spec["restore_backend"],
                                    restore_staging_chunks=group)
        for i in case["host_lru"]:
            slot, why = self.host.allocate(self._no_spill)
            if why != "ok":
                raise StateError(f"host slot for chunk {i}: {why}")
            self.host.slot_view(slot).copy_(bank.data[i])
            self.host.insert(self.keys[i], bank.idents[i], slot)
        for i in case["disk_lru"]:
            ok, why = self.disk.put(self.keys[i], bank.headers[i], bank.data[i].numpy())
            if (ok, why) != (True, "written"):
                raise StateError(f"disk file for chunk {i}: {why}")
        self.expected = sp.expected_structure(case, [k.hex() for k in self.keys], self.file_bytes)
        self.setup = self.observe()
        without_stats = {k: v for k, v in self.setup.items() if k != "file_stats"}
        if without_stats != self.expected:
            raise StateError(f"prepared state differs from the spec: {without_stats} != {self.expected}")
        self.fingerprint = sp.digest(self.setup)

    @staticmethod
    def _no_spill(*_a):
        raise StateError("benchmark state must never spill")

    # ---- verification
    def observe(self) -> dict:
        h, d = self.host, self.disk
        stats = []
        with os.scandir(d.root) as it:
            for e in sorted(it, key=lambda e: e.name):
                if e.name != MARKER:
                    st = e.stat(follow_symlinks=False)
                    stats.append([e.name, st.st_size, st.st_ino, st.st_mtime_ns])
        return {"host_lru": [k.hex() for k in h.entries], "host_slots": [e.slot for e in h.entries.values()],
                "host_pins": [e.pins for e in h.entries.values()], "host_free": list(h._free),
                "host_capacity": h.slots, "staging_slots": h.staging_slots, "staging_busy": h._staging_busy,
                "disk_lru": [k.hex() for k in d.entries], "disk_sizes": list(d.entries.values()),
                "disk_committed": d.committed_bytes, "disk_tmp": d.tmp_bytes,
                "disk_undeletable": sorted(d.undeletable), "disk_budget": d.budget_bytes,
                "disk_max_entries": d.max_entries, "dir_names": [s[0] for s in stats], "file_stats": stats}

    def check(self) -> str:
        """The structural fingerprint; raises StateError if the state is not the prepared one."""
        obs = self.observe()
        if obs != self.setup:
            diff = {k: (obs[k], self.setup[k]) for k in obs if obs[k] != self.setup[k]}
            raise StateError(f"case {self.case['case_id']}: state differs from setup: {diff}")
        return sp.digest(obs)

    def content_check(self) -> dict:
        """SHA-256 of every indexed host slot and disk file, against the bank (reads every file)."""
        host = {k.hex(): _sha(self.host.slot_view(e.slot).numpy()) for k, e in self.host.entries.items()}
        disk = {}
        for k in self.disk.entries:
            with open(self.disk.root / f"{k.hex()}.kvc", "rb") as f:
                disk[k.hex()] = hashlib.sha256(f.read()).hexdigest()
        want_host = {self.keys[i].hex(): self.bank.payload_sha256[i] for i in self.case["host_lru"]}
        want_disk = {self.keys[i].hex(): self.bank.file_sha256[i] for i in self.case["disk_lru"]}
        return {"host": host, "disk": disk, "ok": host == want_host and disk == want_disk}

    # ---- reset
    def reset(self, stored: list) -> None:
        """Undo one sample: drop its admissions, re-establish both LRU orders, then verify the structure."""
        for i in reversed(stored):
            self.host.discard(self.keys[i])
        for i in self.case["host_lru"]:
            with self.host.lease(self.keys[i]):
                pass
        for i in self.case["disk_lru"]:
            ok, why = self.disk.put(self.keys[i], self.bank.headers[i], self.bank.data[i].numpy())
            if (ok, why) != (True, "exists"):
                raise StateError(f"disk recency reset for chunk {i}: {why}")
        self.check()

    def close(self) -> list:
        self.host.close()
        left = self.disk.clear()
        if left:
            raise StateError(f"undeletable files left in {self.root}: {left}")
        shutil.rmtree(self.root)
        return left
