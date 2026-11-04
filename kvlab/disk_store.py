"""Bounded, project-owned disk chunk store (docs/design.md §6).

- Root must be a dedicated directory (marker file). An existing root is fully
  validated first, without following symlinks: the root itself, the marker
  (must be a regular file), and every entry's name and type. Only then are
  stale files deleted, and a missing marker is created with O_EXCL|O_NOFOLLOW.
  An existing marker is never rewritten. Unknown entries -> refuse.
- Filenames are generated from keys only; every path is checked to be a direct
  child of the resolved root before it is opened or unlinked.
- Budget = live entry bytes + reserved temp-file bytes + bytes of files that
  could not be deleted ("undeletable"). Bytes are released only after the file
  is gone; a file that is already missing counts as gone.
- Writes: same-directory temp file + os.replace; temp removed in `finally`.
  No fsync: no durability claim. Reads may be served by the OS page cache.
- Stale files from a previous run are deleted at startup (no recovery).
`write_fn` / `readinto_fn` are the controlled I/O boundary for fault injection;
tests also inject failures at `os.replace` / `os.unlink`.
Scope: one process, sequential calls, a trusted directory. This is not a
defence against concurrent or adversarial changes to the directory.
"""

import errno
import os
import re
import stat
from collections import Counter, OrderedDict
from pathlib import Path

from kvlab import chunk_format as cf
from kvlab.chunk_format import ChunkReadError
from kvlab.identity import ChunkIdentity, Namespace

MARKER = ".kvlab-disk-cache-v1"
MARKER_TEXT = b"kvlab disk cache v1; files here are owned and deleted by kvlab\n"
_FINAL_RE = re.compile(r"^[0-9a-f]{64}\.kvc$")
_TMP_RE = re.compile(r"^[0-9a-f]{64}\.kvc\.tmp[0-9]+$")


def _errname(e: OSError) -> str:
    return errno.errorcode.get(e.errno, str(e.errno))


def _own_name(name: str) -> bool:
    return bool(_FINAL_RE.match(name) or _TMP_RE.match(name))


class DiskStore:
    def __init__(self, root, budget_bytes: int, max_entries: int, write_fn=cf.default_write,
                 readinto_fn=cf.default_readinto):
        if budget_bytes < 0 or max_entries < 0:
            raise ValueError("budget_bytes and max_entries must be >= 0")
        if Path(root).is_symlink():
            raise ValueError(f"{root} is a symlink; the cache root must be a plain directory")
        self.root = Path(root).resolve()
        self.budget_bytes, self.max_entries = budget_bytes, max_entries
        self.write_fn, self.readinto_fn = write_fn, readinto_fn
        self.entries: OrderedDict[bytes, int] = OrderedDict()  # readable entries: key -> file size, LRU first
        self.undeletable: dict[str, int] = {}  # file name -> accounted bytes; never read, never reused
        self.committed_bytes = self.tmp_bytes = self.peak_bytes = 0  # committed = readable entries only
        self.evictions = self.stale_removed = 0
        self.unlink_failures: Counter = Counter()  # "<context>:<errno>" -> count
        self._tmp_counter = 0
        self._open_root()

    # ---- directory ownership
    def _open_root(self):
        try:
            st = os.lstat(self.root)
        except FileNotFoundError:
            self.root.mkdir(parents=True)
            self._create_marker()
            return
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"{self.root} is not a plain directory")
        has_marker, stale = self._validate_existing_root()  # raises before anything is touched
        for name in stale:  # no restart recovery: drop our own stale files (unlink never follows symlinks)
            os.unlink(self._child(name))  # an OSError here aborts construction; nothing is accounted yet
            self.stale_removed += 1
        if not has_marker:
            self._create_marker()

    def _validate_existing_root(self) -> tuple[bool, list[str]]:
        """Accept an empty directory, or one with a regular-file marker and only
        our own file names, each a regular file or a symlink (a symlink is
        unlinked, never followed). Anything else raises ValueError."""
        with os.scandir(self.root) as it:
            found = {e.name: e for e in it}
        if not found:
            return False, []
        marker = found.pop(MARKER, None)
        if marker is None:
            raise ValueError(f"{self.root} is not empty and not a kvlab disk cache; refusing to use it")
        if not marker.is_file(follow_symlinks=False):  # False for symlinks, directories, FIFOs, ...
            raise ValueError(f"{self.root / MARKER} is not a regular file; refusing to use {self.root}")
        unknown = [n for n in found if not _own_name(n)]
        if unknown:
            raise ValueError(f"{self.root} contains unknown entries {sorted(unknown)[:5]}; refusing to use it")
        bad_type = [n for n, e in found.items() if not (e.is_file(follow_symlinks=False) or e.is_symlink())]
        if bad_type:
            raise ValueError(f"{self.root} has cache-named entries that are not files {sorted(bad_type)[:5]}; "
                             "refusing to use it")
        return True, sorted(found)

    def _create_marker(self):
        fd = os.open(self.root / MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(MARKER_TEXT)

    def _child(self, name: str) -> Path:
        if not _own_name(name):
            raise ValueError(f"refusing non-cache filename {name!r}")
        path = self.root / name
        if path.parent != self.root:
            raise ValueError(f"{path} escapes {self.root}")
        return path

    def _final_path(self, key: bytes) -> Path:
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        return self._child(f"{key.hex()}.kvc")

    # ---- accounting
    @property
    def undeletable_bytes(self) -> int:
        return sum(self.undeletable.values())

    @property
    def used_bytes(self) -> int:
        return self.committed_bytes + self.tmp_bytes + self.undeletable_bytes

    @property
    def file_count(self) -> int:
        """Tracked files bounded by max_entries: readable entries + undeletable files."""
        return len(self.entries) + len(self.undeletable)

    def _note_peak(self):
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        if self.used_bytes > self.budget_bytes:
            raise AssertionError(f"disk budget exceeded: {self.used_bytes} > {self.budget_bytes}")

    def __contains__(self, key: bytes) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    # ---- deletion
    def _unlink(self, name: str, context: str) -> str | None:
        """Delete one of our files. Returns None once it is gone (already missing
        counts as gone), or the errno name of an OSError, which is recorded.
        Any other exception propagates."""
        path = self._child(name)
        try:
            os.unlink(path)  # never follows symlinks
        except FileNotFoundError:
            return None
        except OSError as e:
            err = _errname(e)
            self.unlink_failures[f"{context}:{err}"] += 1
            return err
        return None

    def _remove(self, key: bytes, context: str) -> str | None:
        """Drop a readable entry. Its bytes are released only if the file is
        gone. Otherwise the file is tracked as undeletable: still in the budget,
        never read, never retried automatically. Returns the errno name or None."""
        name = self._final_path(key).name
        err = self._unlink(name, context)
        size = self.entries.pop(key)
        self.committed_bytes -= size
        if err is not None:
            self.undeletable[name] = size
        return err

    # ---- write
    def put(self, key: bytes, header: bytes, payload) -> tuple[bool, str]:
        """Write one chunk file. Returns (ok, reason). Expected filesystem
        failures (OSError, including a failed eviction) give (False, reason),
        i.e. a skipped cache write; any other exception propagates.
        The temp file never outlives this call unless it cannot be deleted,
        in which case its reservation is kept as undeletable bytes."""
        if key in self.entries:
            self.entries.move_to_end(key)
            return True, "exists"
        final = self._final_path(key)
        size = len(header) + memoryview(payload).nbytes + cf.U32.size
        if size > self.budget_bytes or self.max_entries == 0:
            return False, "oversized_for_disk"
        if final.name in self.undeletable:
            return False, "undeletable_file_for_key"  # an earlier file for this key could not be deleted
        # Eviction cannot free reserved temp bytes or undeletable files: give up before evicting anything.
        if (self.tmp_bytes + self.undeletable_bytes + size > self.budget_bytes
                or len(self.undeletable) >= self.max_entries):
            return False, "disk_budget_unavailable"
        while self.used_bytes + size > self.budget_bytes or self.file_count >= self.max_entries:
            err = self._remove(next(iter(self.entries)), "evict")  # entries is non-empty by the check above
            if err is not None:
                return False, f"evict_oserror:{err}"  # bounded: this victim is now undeletable, not retried
            self.evictions += 1
        self._tmp_counter += 1
        tmp = self._child(f"{final.name}.tmp{self._tmp_counter}")
        self.tmp_bytes += size  # reserved before the temp file exists
        self._note_peak()
        committed = False
        try:
            with open(tmp, "xb", buffering=0) as f:  # O_CREAT|O_EXCL: never opens an existing path or symlink
                cf.write_chunk(f, header, payload, self.write_fn)
            os.replace(tmp, final)
            committed = True
            self.entries[key] = size
            self.committed_bytes += size
            return True, "written"
        except OSError as e:
            return False, f"write_oserror:{_errname(e)}"
        finally:
            self.tmp_bytes -= size
            if not committed and self._unlink(tmp.name, "tmp") is not None:
                self.undeletable[tmp.name] = size  # the temp file is still there: keep its reservation

    # ---- read
    def read_into(self, key: bytes, ns: Namespace, ident: ChunkIdentity, dest) -> tuple[bool, str, int]:
        """Validate and read a chunk payload into `dest`. Returns (ok, reason, file_bytes).
        Known entry problems are misses; the bad entry is removed (or, if its
        file cannot be deleted, kept only as undeletable bytes, never read)."""
        if key not in self.entries:
            return False, "disk_absent", 0
        path = self._final_path(key)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, "rb", buffering=0) as f:
                size = os.fstat(f.fileno()).st_size
                cf.read_chunk_into(f, size, ns, ident, dest, self.readinto_fn)
        except FileNotFoundError:
            reason = "disk_file_missing"
        except ChunkReadError as e:
            reason = f"disk_{e.reason}"
        except OSError as e:
            reason = f"disk_read_oserror:{_errname(e)}"
        else:
            self.entries.move_to_end(key)
            return True, "hit", size
        self._remove(key, "invalid_entry")
        return False, reason, 0

    def clear(self) -> list[str]:
        """Delete every file this store tracks: one attempt per readable entry,
        and one more explicit attempt per earlier undeletable file. Returns the
        names that still could not be deleted (their bytes stay accounted)."""
        earlier = list(self.undeletable)
        for key in list(self.entries):
            self._remove(key, "clear")
        for name in earlier:
            if self._unlink(name, "clear") is None:
                del self.undeletable[name]
        return sorted(self.undeletable)

    def scan_dir_bytes(self) -> tuple[int, list[str]]:
        """Actual bytes of cache files in the directory, symlinks not followed (for budget tests)."""
        total, names = 0, []
        with os.scandir(self.root) as it:
            for e in it:
                if e.name != MARKER:
                    total += e.stat(follow_symlinks=False).st_size
                    names.append(e.name)
        return total, sorted(names)
