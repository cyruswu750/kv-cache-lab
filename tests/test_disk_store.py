import errno
import os

import pytest

from kvlab import chunk_format as cf
from kvlab import disk_store as ds_mod
from kvlab.disk_store import MARKER, DiskStore
from m1_toy import deny_unlink, ident, payload, tokens, toy_ns

NS = toy_ns()
FILE = cf.file_size(NS)


def entry(i, seed=None):
    idt = ident(NS, tokens(4 * (i + 1)), i)
    return idt, cf.encode_header(NS, idt), payload(NS, seed=i if seed is None else seed)


def store(tmp_path, n_files=2, **kw):
    return DiskStore(tmp_path / "cache", budget_bytes=n_files * FILE, max_entries=kw.pop("max_entries", 100), **kw)


def read(d, idt):
    dest = bytearray(cf.payload_len(NS))
    ok, why, n = d.read_into(idt.key, NS, idt, dest)
    return ok, why, bytes(dest)


def test_round_trip_and_lru_eviction(tmp_path):
    d = store(tmp_path, n_files=2)
    e0, e1, e2 = entry(0), entry(1), entry(2)
    for idt, hdr, data in (e0, e1):
        assert d.put(idt.key, hdr, data) == (True, "written")
    assert read(d, e0[0])[:2] == (True, "hit")  # touch e0 -> e1 is LRU
    assert d.put(e2[0].key, e2[1], e2[2]) == (True, "written")
    assert e1[0].key not in d and e0[0].key in d and d.evictions == 1
    ok, _, data = read(d, e2[0])
    assert ok and data == bytes(e2[2])
    total, names = d.scan_dir_bytes()
    assert total == d.committed_bytes == 2 * FILE <= d.budget_bytes and len(names) == 2
    assert d.put(e0[0].key, e0[1], e0[2]) == (True, "exists")


def test_temp_file_counts_against_budget_during_write(tmp_path):
    seen = []

    def watching_write(f, mv):
        total, names = d.scan_dir_bytes()
        seen.append((d.tmp_bytes, d.used_bytes, any(".tmp" in n for n in names), total))
        assert d.used_bytes <= d.budget_bytes and total <= d.budget_bytes
        return f.write(mv)
    d = store(tmp_path, n_files=2, write_fn=watching_write)
    for i in range(4):
        idt, hdr, data = entry(i)
        assert d.put(idt.key, hdr, data)[0]
    assert all(tmp == FILE and has_tmp for tmp, _, has_tmp, _ in seen)
    assert d.tmp_bytes == 0 and d.peak_bytes == 2 * FILE


def test_enospc_skips_write_and_cleans_temp(tmp_path):
    calls = []

    def full_disk(f, mv):
        calls.append(1)
        if len(calls) > 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return f.write(mv[:10])
    d = store(tmp_path, write_fn=full_disk)
    idt, hdr, data = entry(0)
    assert d.put(idt.key, hdr, data) == (False, "write_oserror:ENOSPC")
    assert d.scan_dir_bytes() == (0, []) and d.tmp_bytes == 0 and d.committed_bytes == 0 and len(d) == 0


def test_replace_failure_cleans_temp(tmp_path, monkeypatch):
    d = store(tmp_path)

    def bad_replace(a, b):
        raise OSError(errno.EXDEV, "cross-device")
    monkeypatch.setattr(ds_mod.os, "replace", bad_replace)
    idt, hdr, data = entry(0)
    assert d.put(idt.key, hdr, data) == (False, "write_oserror:EXDEV")
    assert d.scan_dir_bytes() == (0, []) and d.used_bytes == 0


def test_non_oserror_propagates_and_still_cleans_temp(tmp_path):
    def buggy(f, mv):
        raise TypeError("bug")
    d = store(tmp_path, write_fn=buggy)
    idt, hdr, data = entry(0)
    with pytest.raises(TypeError):
        d.put(idt.key, hdr, data)
    assert d.scan_dir_bytes() == (0, []) and d.used_bytes == 0


def test_oversized_and_entry_limit(tmp_path):
    d = DiskStore(tmp_path / "c", budget_bytes=FILE - 1, max_entries=10)
    idt, hdr, data = entry(0)
    assert d.put(idt.key, hdr, data) == (False, "oversized_for_disk")
    d2 = DiskStore(tmp_path / "c2", budget_bytes=10 * FILE, max_entries=1)
    for i in range(3):
        idt, hdr, data = entry(i)
        assert d2.put(idt.key, hdr, data)[0]
    assert len(d2) == 1 and d2.evictions == 2


@pytest.mark.parametrize("damage, expected", [
    ("delete", "disk_file_missing"),
    ("truncate", "disk_size_mismatch"),
    ("flip_payload", "disk_payload_crc"),
    ("flip_header", "disk_header_crc"),
])
def test_read_faults_are_misses_and_remove_entry(tmp_path, damage, expected):
    d = store(tmp_path)
    idt, hdr, data = entry(0)
    d.put(idt.key, hdr, data)
    path = d.root / f"{idt.key.hex()}.kvc"
    raw = bytearray(path.read_bytes())
    if damage == "delete":
        path.unlink()
    elif damage == "truncate":
        path.write_bytes(raw[:-1])
    elif damage == "flip_payload":
        raw[-10] ^= 1
        path.write_bytes(raw)
    else:
        raw[150] ^= 1  # inside token ids
        path.write_bytes(raw)
    assert read(d, idt)[:2] == (False, expected)
    assert idt.key not in d and not path.exists() and d.committed_bytes == 0


def test_read_oserror_is_a_miss(tmp_path):
    def eio(f, mv):
        raise OSError(errno.EIO, "I/O error")
    d = store(tmp_path, readinto_fn=eio)
    idt, hdr, data = entry(0)
    d.put(idt.key, hdr, data)
    assert read(d, idt)[:2] == (False, "disk_read_oserror:EIO")


def test_refuses_foreign_directories_and_deletes_nothing(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError):
        DiskStore(foreign, budget_bytes=FILE, max_entries=1)
    assert (foreign / "notes.txt").read_text() == "keep me"
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / MARKER).write_text("x")
    (owned / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError):
        DiskStore(owned, budget_bytes=FILE, max_entries=1)
    assert (owned / "notes.txt").exists()
    link = tmp_path / "link"
    link.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError):
        DiskStore(link, budget_bytes=FILE, max_entries=1)


def test_stale_cleanup_only_touches_own_files_and_never_follows_symlinks(tmp_path):
    outside = tmp_path / "outside.kvc"
    outside.write_text("outside target")
    sibling = tmp_path / ("a" * 64 + ".kvc")
    sibling.write_text("sibling outside the root")
    root = tmp_path / "cache"
    root.mkdir()
    (root / MARKER).write_text("x")
    (root / ("b" * 64 + ".kvc")).write_bytes(b"stale")
    (root / ("c" * 64 + ".kvc.tmp7")).write_bytes(b"stale tmp")
    os.symlink(outside, root / ("d" * 64 + ".kvc"))
    d = DiskStore(root, budget_bytes=FILE, max_entries=1)
    assert d.stale_removed == 3 and d.scan_dir_bytes() == (0, [])
    assert outside.read_text() == "outside target" and sibling.exists()


def test_child_paths_validated(tmp_path):
    d = store(tmp_path)
    for bad in ["../x.kvc", "/etc/passwd", "a" * 63 + ".kvc", "A" * 64 + ".kvc", "x"]:
        with pytest.raises(ValueError):
            d._child(bad)
    with pytest.raises(ValueError):
        d._final_path(b"short")


# ---- M1.1 regressions (external review): failed deletion must keep its bytes accounted

def test_eviction_unlink_failure_is_a_bounded_write_skip(tmp_path, monkeypatch):
    """Review repro: 8-byte budget; the eviction victim cannot be unlinked (EACCES)."""
    d = DiskStore(tmp_path / "c", budget_bytes=8, max_entries=10)
    ka, kb, kc = b"\xaa" * 32, b"\xbb" * 32, b"\xcc" * 32
    assert d.put(ka, b"", b"1234") == (True, "written")  # 4 B payload + 4 B CRC
    a_name = f"{ka.hex()}.kvc"
    attempts = deny_unlink(monkeypatch, lambda n: n == a_name)
    assert d.put(kb, b"", b"5678") == (False, "evict_oserror:EACCES")  # reported write skip, no exception
    assert attempts == [a_name]
    assert d.scan_dir_bytes() == (8, [a_name])  # the file is still there ...
    assert d.used_bytes == 8 <= d.budget_bytes  # ... so its bytes are still accounted
    assert (d.committed_bytes, d.tmp_bytes, d.undeletable) == (0, 0, {a_name: 8})
    assert ka not in d and kb not in d and d.evictions == 0
    assert d.unlink_failures == {"evict:EACCES": 1}
    # Later writes neither retry the same victim nor exceed the budget.
    assert d.put(kc, b"", b"9abc") == (False, "disk_budget_unavailable")
    assert d.put(kc, b"", b"9abc") == (False, "disk_budget_unavailable")
    assert attempts == [a_name] and d.used_bytes == 8 and d.scan_dir_bytes() == (8, [a_name])
    monkeypatch.undo()  # fault gone: one explicit clear() attempt releases it
    assert d.clear() == [] and (d.used_bytes, d.undeletable) == (0, {}) and d.scan_dir_bytes() == (0, [])
    assert d.put(kc, b"", b"9abc") == (True, "written")


def test_eviction_failure_keeps_other_entries_and_moves_on_to_the_next_victim(tmp_path, monkeypatch):
    d = store(tmp_path, n_files=2)
    e0, e1, e2, e3 = (entry(i) for i in range(4))
    for idt, hdr, data in (e0, e1):
        assert d.put(idt.key, hdr, data) == (True, "written")
    n0 = f"{e0[0].key.hex()}.kvc"
    attempts = deny_unlink(monkeypatch, lambda n: n == n0)
    assert d.put(e2[0].key, e2[1], e2[2]) == (False, "evict_oserror:EACCES")
    assert d.scan_dir_bytes()[0] == d.used_bytes == 2 * FILE  # accounting matches the directory
    assert read(d, e1[0])[:2] == (True, "hit")  # the other entry is untouched and still readable
    assert read(d, e0[0])[:2] == (False, "disk_absent")  # the undeletable file is never read
    assert d.put(e3[0].key, e3[1], e3[2]) == (True, "written")  # evicts e1, does not retry e0
    assert attempts == [n0] and e1[0].key not in d and d.evictions == 1
    total, names = d.scan_dir_bytes()
    assert total == d.used_bytes == 2 * FILE <= d.budget_bytes and n0 in names and len(names) == 2
    ok, _, data = read(d, e3[0])
    assert ok and data == bytes(e3[2])


def test_undeletable_files_count_against_max_entries(tmp_path, monkeypatch):
    d = DiskStore(tmp_path / "c", budget_bytes=10 * FILE, max_entries=1)
    e0, e1, e2 = (entry(i) for i in range(3))
    assert d.put(e0[0].key, e0[1], e0[2]) == (True, "written")
    attempts = deny_unlink(monkeypatch, lambda n: n.startswith(e0[0].key.hex()))
    assert d.put(e1[0].key, e1[1], e1[2]) == (False, "evict_oserror:EACCES")
    assert d.put(e2[0].key, e2[1], e2[2]) == (False, "disk_budget_unavailable")
    assert len(attempts) == 1 and d.file_count == 1 and len(d.scan_dir_bytes()[1]) == 1


@pytest.mark.parametrize("fault", ["flip_payload", "read_eio"])
def test_invalid_entry_cleanup_failure_is_a_safe_miss(tmp_path, monkeypatch, fault):
    def eio(f, mv):
        raise OSError(errno.EIO, "I/O error")
    d = store(tmp_path, n_files=2, **({"readinto_fn": eio} if fault == "read_eio" else {}))
    idt, hdr, data = entry(0)
    assert d.put(idt.key, hdr, data) == (True, "written")
    path = d.root / f"{idt.key.hex()}.kvc"
    if fault == "flip_payload":
        raw = bytearray(path.read_bytes())
        raw[-10] ^= 1
        path.write_bytes(raw)
    damaged = path.read_bytes()
    attempts = deny_unlink(monkeypatch, lambda n: n == path.name)
    expected = "disk_payload_crc" if fault == "flip_payload" else "disk_read_oserror:EIO"
    assert read(d, idt)[:2] == (False, expected)  # a miss (safe fallback), not an exception
    assert idt.key not in d and attempts == [path.name]
    assert path.read_bytes() == damaged  # the file could not be deleted ...
    assert d.used_bytes == d.undeletable_bytes == d.scan_dir_bytes()[0] == FILE  # ... and stays accounted
    assert d.committed_bytes == 0 and d.unlink_failures == {"invalid_entry:EACCES": 1}
    assert read(d, idt)[:2] == (False, "disk_absent")  # invalid data is never read again
    assert d.put(idt.key, hdr, data) == (False, "undeletable_file_for_key")  # nor reused under its key
    assert path.read_bytes() == damaged and attempts == [path.name]
    e1 = entry(1)
    assert d.put(e1[0].key, e1[1], e1[2]) == (True, "written")  # the remaining budget is still usable
    assert d.scan_dir_bytes()[0] == d.used_bytes == 2 * FILE <= d.budget_bytes


@pytest.mark.parametrize("failure", ["enospc", "replace_exdev", "non_oserror"])
def test_temp_cleanup_failure_keeps_the_reservation(tmp_path, monkeypatch, failure):
    def failing_write(f, mv):
        if failure == "non_oserror":
            raise TypeError("bug in writer")
        f.write(mv[:10])
        raise OSError(errno.ENOSPC, "No space left on device")
    def failing_replace(src, dst):
        raise OSError(errno.EXDEV, "cross-device")
    real_replace = os.replace
    if failure == "replace_exdev":
        monkeypatch.setattr(os, "replace", failing_replace)
    d = store(tmp_path, n_files=2, **({} if failure == "replace_exdev" else {"write_fn": failing_write}))
    attempts = deny_unlink(monkeypatch, lambda n: ".kvc.tmp" in n)
    idt, hdr, data = entry(0)
    if failure == "non_oserror":
        with pytest.raises(TypeError, match="bug in writer"):  # programming errors still propagate
            d.put(idt.key, hdr, data)
    else:
        expected = "write_oserror:ENOSPC" if failure == "enospc" else "write_oserror:EXDEV"
        assert d.put(idt.key, hdr, data) == (False, expected)
    assert len(attempts) == 1
    total, names = d.scan_dir_bytes()
    assert names == attempts and total <= FILE  # the temp file is still on disk
    assert (d.tmp_bytes, d.committed_bytes, d.undeletable_bytes, d.used_bytes) == (0, 0, FILE, FILE)
    assert d.unlink_failures == {"tmp:EACCES": 1} and len(d) == 0
    # The kept reservation still bounds later writes.
    d.write_fn = cf.default_write
    monkeypatch.setattr(os, "replace", real_replace)
    e1, e2 = entry(1), entry(2)
    assert d.put(e1[0].key, e1[1], e1[2]) == (True, "written")
    assert d.put(e2[0].key, e2[1], e2[2]) == (True, "written")  # needs an eviction: e1, not the temp file
    assert d.evictions == 1 and e1[0].key not in d and len(attempts) == 1
    total, names = d.scan_dir_bytes()
    assert total <= d.used_bytes == 2 * FILE <= d.budget_bytes and attempts[0] in names


def test_stale_file_that_cannot_be_deleted_aborts_construction(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    (root / MARKER).write_text("x")
    stale = root / ("b" * 64 + ".kvc")
    stale.write_bytes(b"stale")
    deny_unlink(monkeypatch, lambda n: n == stale.name)
    with pytest.raises(PermissionError):  # before any request; no store object exists
        DiskStore(root, budget_bytes=FILE, max_entries=1)
    assert stale.read_bytes() == b"stale"


# ---- M1.1 regressions (external review): the marker must never be written through a link

@pytest.mark.parametrize("target", ["existing", "dangling"])
def test_marker_symlink_is_refused_before_anything_is_written_or_deleted(tmp_path, target):
    external = tmp_path / "external.txt"
    if target == "existing":
        external.write_text("KEEP")
    root = tmp_path / "cache"
    root.mkdir()
    os.symlink(external, root / MARKER)
    cached = root / ("b" * 64 + ".kvc")
    cached.write_bytes(b"existing cache file")
    tmp_file = root / ("c" * 64 + ".kvc.tmp3")
    tmp_file.write_bytes(b"existing temp file")
    before = sorted(os.listdir(root))
    with pytest.raises(ValueError, match="not a regular file"):
        DiskStore(root, budget_bytes=FILE, max_entries=4)
    if target == "existing":
        assert external.read_text() == "KEEP"
    else:
        assert not os.path.lexists(external)  # nothing created through the dangling link
    assert (root / MARKER).is_symlink() and os.readlink(root / MARKER) == str(external)
    assert cached.read_bytes() == b"existing cache file" and tmp_file.read_bytes() == b"existing temp file"
    assert sorted(os.listdir(root)) == before


def test_non_regular_marker_or_entry_is_refused_and_nothing_is_deleted(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / MARKER).mkdir()  # a directory, not a regular file
    cached = root / ("b" * 64 + ".kvc")
    cached.write_bytes(b"keep")
    with pytest.raises(ValueError, match="not a regular file"):
        DiskStore(root, budget_bytes=FILE, max_entries=4)
    assert cached.read_bytes() == b"keep"
    root2 = tmp_path / "cache2"
    root2.mkdir()
    (root2 / MARKER).write_text("x")
    (root2 / ("e" * 64 + ".kvc")).mkdir()  # cache-named directory
    cached2 = root2 / ("b" * 64 + ".kvc")
    cached2.write_bytes(b"keep")
    with pytest.raises(ValueError, match="not files"):
        DiskStore(root2, budget_bytes=FILE, max_entries=4)
    assert cached2.read_bytes() == b"keep" and (root2 / ("e" * 64 + ".kvc")).is_dir()


def test_an_existing_marker_is_never_rewritten_and_a_new_one_is_a_plain_file(tmp_path):
    external = tmp_path / "external.txt"
    external.write_text("KEEP")
    root = tmp_path / "linked"
    root.mkdir()
    os.link(external, root / MARKER)  # a regular file that happens to be a hard link to outside data
    DiskStore(root, budget_bytes=FILE, max_entries=1)
    assert external.read_text() == "KEEP" and (root / MARKER).read_text() == "KEEP"
    d = DiskStore(tmp_path / "fresh", budget_bytes=FILE, max_entries=1)
    m = d.root / MARKER
    assert m.is_file() and not m.is_symlink() and m.read_text().startswith("kvlab disk cache v1")


def test_symlinked_chunk_file_is_not_followed_on_read(tmp_path):
    d = store(tmp_path)
    idt, hdr, data = entry(0)
    d.put(idt.key, hdr, data)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside data")
    path = d.root / f"{idt.key.hex()}.kvc"
    path.unlink()
    os.symlink(outside, path)
    assert read(d, idt)[:2] == (False, "disk_read_oserror:ELOOP")
    assert not os.path.lexists(path) and outside.read_bytes() == b"outside data" and d.used_bytes == 0
