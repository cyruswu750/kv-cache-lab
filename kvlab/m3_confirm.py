"""M3_CONFIRM_V1: premeasurement freeze, budgeted batch execution, and their checks
(docs/M3_CONFIRM_PROTOCOL.md). CPU only; this module never imports torch.

Three records bind an evidence directory to the plan:

1. **Freeze manifest** (`freeze()`, written once, after review and before measurement):
   - the base Git HEAD;
   - the SHA-256 of `git diff HEAD --binary`, with the diff itself saved as a sidecar file;
   - the actual bytes (SHA-256) of every file that a session records about itself, i.e. what
     `kvlab.m3_bench.source_identity` hashes, untracked new files included;
   - every other untracked, non-ignored file outside the confirmation results;
   - the hashes of the protocol documents;
   - the exact session commands, generated from the spec;
   - the batch budget;
   - the model and software stack, which must equal the spec's required stack. A different
     stack is refused, never substituted.

   The manifest never hashes itself, and it lives outside the run directory.
2. **Batch record** (`run_batch()`, in the run directory): written atomically after every state
   change. It holds the manifest's SHA-256, the pre-launch verification, every launched
   command, its exit code and timing, the SHA-256 of every session file, the execution status
   and the reason the batch stopped. It records execution only; evidence completeness is
   decided by the analysis.
3. **Session files**, written by the unchanged `scripts/m3_measure.py`.

`kvlab/m3_validate.py` checks all three against each other and against the spec.
"""

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

MANIFEST_VERSION = 1
BATCH_RECORD_VERSION = 2  # 2: execution status and stop reason, separate from evidence completeness
BATCH_RECORD = "batch_record.json"
PROTOCOL_DOC = "docs/M3_CONFIRM_PROTOCOL.md"
DOCUMENTS = (PROTOCOL_DOC, "docs/M3_PROTOCOL.md", "docs/M3_DESIGN.md", "requirements.txt")
# files whose current bytes must still equal the freeze when the analysis runs
ANALYSIS_FILES = ("kvlab/m3_validate.py", "kvlab/m3_confirm.py", "kvlab/m3_stats.py", "kvlab/perf_stats.py",
                  "kvlab/perf.py", "kvlab/m3_bench.py", "scripts/m3_analyze.py")
DRAFT_MARKER = "DRAFT"
PACKAGE_PREFIXES = ("nvidia", "torch", "transformers", "numpy", "safetensors", "tokenizers", "huggingface",
                    "triton", "accelerate")  # the filter of kvlab.diag_harness.package_versions


def finite(x) -> bool:
    """A real JSON number (not a bool) that converts to a finite float. NaN, Infinity and an
    integer too large for a float are not finite: time comparisons, arithmetic and formatting
    mix ints with floats, and converting such an integer raises OverflowError."""
    if type(x) is int:
        try:
            x = float(x)
        except OverflowError:
            return False
    return type(x) is float and math.isfinite(x)


def _reject_non_finite(name):
    raise ValueError(f"non-finite number {name}")


def load_json_finite(data):
    """json.loads that rejects the NaN, Infinity and -Infinity literals, which the stdlib accepts
    by default. A number literal beyond the float range (e.g. 1e400) still parses to inf; every
    time compared later must pass finite(), which rejects it."""
    return json.loads(data, parse_constant=_reject_non_finite)


class Refused(Exception):
    """A freeze or launch precondition does not hold; `problems` lists every failed check."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(root, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout


# ---- identity (mirrors kvlab.m3_bench.source_identity and kvlab.diag_harness, without torch)
def source_identity(root: Path) -> dict:
    """The same record as `kvlab.m3_bench.source_identity` (a test checks the equality)."""
    root = Path(root)
    files = sorted([*root.glob("kvlab/*.py"), *root.glob("scripts/*.py"), *root.glob("tests/**/*.py"),
                    root / "pyproject.toml"])
    diff = _git(root, "diff", "HEAD", "--binary")
    return {"git_head": _git(root, "rev-parse", "HEAD").strip(),
            "git_status_porcelain": _git(root, "status", "--porcelain").splitlines(),
            "git_diff_head_sha256": sha256_bytes(diff.encode()),
            "files_sha256": {str(p.relative_to(root)): sha256_bytes(p.read_bytes()) for p in files}}


def package_versions() -> dict:
    out = {}
    for d in importlib.metadata.distributions():
        name = d.metadata["Name"]
        if name and name.lower().startswith(PACKAGE_PREFIXES):
            out[name] = d.version
    return dict(sorted(out.items()))


def weight_blobs(root: Path, model_id: str, revision: str) -> dict:
    snap = Path(root) / ".hf_cache" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not snap.is_dir():
        return {"error": f"no snapshot dir {snap}"}
    return {p.name: {"blob": os.path.basename(os.readlink(p)) if p.is_symlink() else None, "bytes": p.stat().st_size}
            for p in sorted(snap.iterdir())}


def gpu_identity() -> dict:
    """Read-only nvidia-smi query; it does not initialise CUDA in this process."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        name, driver = (x.strip() for x in out[0].split(","))
        return {"gpu_name": name, "driver_version": driver}
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError) as e:
        return {"gpu_name": None, "driver_version": None, "error": repr(e)}


def parse_gpu_static(text) -> dict:
    """name and driver_version from the `gpu_static` CSV a session records (perf_harness.environment)."""
    try:
        name, driver = (x.strip() for x in str(text).splitlines()[1].split(",")[:2])
        return {"gpu_name": name, "driver_version": driver}
    except (IndexError, ValueError):
        return {"gpu_name": None, "driver_version": None}


def cpu_model():
    try:
        return next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                     if ln.startswith("model name")), None)
    except OSError:
        return None


def current_stack(root: Path, spec: dict) -> dict:
    m = spec["required_model"]
    return {"python": platform.python_version(), "packages": package_versions(),
            "weights": weight_blobs(root, m["model_id"], m["revision"]), **gpu_identity(),
            "kernel": platform.release(), "cpu": cpu_model()}


def stack_problems(stack: dict, spec: dict) -> list[str]:
    req, out = spec["required_stack"], []
    if stack.get("python") != req["python"]:
        out.append(f"python {stack.get('python')!r}, required {req['python']!r}")
    if stack.get("packages") != spec["required_packages"]:
        got, want = stack.get("packages") or {}, spec["required_packages"]
        diff = {k: (got.get(k), want.get(k)) for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)}
        out.append(f"packages differ from the required stack (found, required): {diff}")
    for k in ("gpu_name", "driver_version"):
        if stack.get(k) != req[k]:
            out.append(f"{k} {stack.get(k)!r}, required {req[k]!r}")
    w = (stack.get("weights") or {}).get("model.safetensors") or {}
    if w.get("blob") != req["weights_blob"] or w.get("bytes") != req["weights_bytes"]:
        out.append(f"model.safetensors {w!r}, required blob {req['weights_blob']} ({req['weights_bytes']} B)")
    return out


def model_constant_problems(root: Path, spec: dict) -> list[str]:
    """kvlab/model.py pins what the sessions load; read as text (no torch import)."""
    text = (Path(root) / "kvlab" / "model.py").read_text()
    m = spec["required_model"]
    out = []
    for const, want in (("MODEL_ID", m["model_id"]), ("MODEL_REVISION", m["revision"])):
        got = re.search(rf'^{const} = "([^"]*)"', text, re.M)
        if not got or got.group(1) != want:
            out.append(f"kvlab/model.py {const} {got and got.group(1)!r}, spec {want!r}")
    if not re.search(r"^DTYPE = torch\.bfloat16\b", text, re.M):
        out.append("kvlab/model.py DTYPE is not torch.bfloat16")
    attn = re.search(r'^ATTN_IMPLEMENTATION = "([^"]*)"', text, re.M)
    if not attn or attn.group(1) != m["attn_implementation"]:
        out.append(f"kvlab/model.py ATTN_IMPLEMENTATION {attn and attn.group(1)!r}, spec {m['attn_implementation']!r}")
    return out


# ---- the plan, generated from the spec (single source of truth)
def case_arg(spec) -> str:
    return ",".join(f"{L}:{S}" for L, S in spec["cases"])


def commands_for(spec: dict) -> list[dict]:
    """The exact session commands, in launch order. Run from the repository root."""
    return [{"session": sid,
             "argv": [spec["python"], "scripts/m3_measure.py", "--out-dir", spec["run_dir"], "--session", sid,
                      "--cases", case_arg(spec), "--blocks", str(spec["blocks"]), "--warmup", str(spec["warmup"]),
                      "--reps", str(spec["reps"]), "--seed", str(seed), "--staging-chunks", str(spec["staging_chunks"])],
             "env": dict(spec["session_env"])}
            for sid, seed in spec["sessions"].items()]


def expected_args(spec: dict, sid: str) -> dict:
    """`vars(args)` as scripts/m3_measure.py records it for that command (meta.args)."""
    return {"out_dir": spec["run_dir"], "session": sid, "cases": case_arg(spec), "blocks": spec["blocks"],
            "warmup": spec["warmup"], "reps": spec["reps"], "seed": spec["sessions"][sid],
            "staging_chunks": spec["staging_chunks"], "profile": False, "profile_reps": 5}


def run_dir_files(spec: dict) -> set:
    """Every file a complete batch leaves in the run directory; anything else is unexpected."""
    out = {BATCH_RECORD}
    for sid in spec["sessions"]:
        out |= {f"session_{sid}.json", f"session_{sid}.gpu_monitor.csv", f"logs/session_{sid}.log"}
    return out


def scratch_dir(root: Path, run_dir: str, sid: str) -> Path:
    """The disk-tier scratch area scripts/m3_measure.py uses for one session."""
    return Path(root) / ".kvcache" / "m3" / Path(run_dir).name / sid


# ---- freeze
def diff_sidecar(spec) -> str:
    return str(Path(spec["freeze_manifest"]).with_suffix(".diff"))


def launch_env_problems(spec: dict, environ=None) -> list[str]:
    """The session environment (inherited + the frozen session_env) must be the M3 one. The
    runner does not record CUDA_VISIBLE_DEVICES, so it must not be set at all."""
    env = {**(os.environ if environ is None else environ), **spec["session_env"]}
    out = [f"environment {k}={env.get(k)!r}, required {v!r}" for k, v in spec["required_process_env"].items()
           if env.get(k) != v]
    if "CUDA_VISIBLE_DEVICES" in env:
        out.append(f"CUDA_VISIBLE_DEVICES is set ({env['CUDA_VISIBLE_DEVICES']!r}); the M3 sessions ran without it")
    return out


def build_manifest(root: Path, spec: dict, *, stack=None, now=None, require_frozen_protocol=True):
    """(manifest, diff_text). Raises Refused if a precondition fails; nothing is written."""
    root = Path(root)
    problems = []
    src = source_identity(root)
    if not re.fullmatch(r"[0-9a-f]{40}", src["git_head"]):
        problems.append(f"no Git HEAD ({src['git_head']!r})")
    changed_results = [ln for ln in _git(root, "diff", "HEAD", "--name-only", "--", "results").splitlines() if ln]
    if changed_results:
        problems.append(f"tracked evidence under results/ differs from HEAD: {changed_results}")
    doc = root / PROTOCOL_DOC
    if not doc.is_file():
        problems.append(f"{PROTOCOL_DOC} is missing")
    elif require_frozen_protocol and DRAFT_MARKER in doc.read_text().splitlines()[0]:
        problems.append(f"{PROTOCOL_DOC} is still marked {DRAFT_MARKER}; finish the review and update its status line first")
    problems += model_constant_problems(root, spec)
    stack = current_stack(root, spec) if stack is None else stack
    problems += stack_problems(stack, spec)
    for rel in (spec["freeze_manifest"], diff_sidecar(spec), spec["run_dir"]):
        if (root / rel).exists():
            problems.append(f"{rel} already exists")
    if problems:
        raise Refused(problems)
    diff = _git(root, "diff", "HEAD", "--binary")
    excluded = Path(spec["freeze_manifest"]).parent.as_posix() + "/"
    untracked = sorted(p for p in _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
                       if p and not p.startswith(excluded))
    now = time.time() if now is None else now
    manifest = {
        "manifest_version": MANIFEST_VERSION, "experiment": spec["name"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), "frozen_unix": now,
        "base_git_head": src["git_head"], "git_status_porcelain": src["git_status_porcelain"],
        "git_diff_head_sha256": src["git_diff_head_sha256"],
        "git_diff_head_name_status": _git(root, "diff", "HEAD", "--name-status").splitlines(),
        "git_diff_file": Path(diff_sidecar(spec)).name,
        "files_sha256": src["files_sha256"],
        "untracked_files_sha256": {p: sha256_file(root / p) for p in untracked if (root / p).is_file()},
        "documents_sha256": {d: sha256_file(root / d) for d in DOCUMENTS},
        "analysis_files": list(ANALYSIS_FILES),
        "stack": stack, "required_model": dict(spec["required_model"]),
        "run_dir": spec["run_dir"], "commands": commands_for(spec), "budget_s": spec["budget_s"],
        "note": "Hashes are of the working tree at freeze time; no commit is required. The manifest does not hash itself.",
    }
    if manifest["git_diff_head_sha256"] != sha256_bytes(diff.encode()):
        raise Refused(["the working tree changed while the manifest was built"])
    return manifest, diff


def freeze(root: Path, spec: dict, **kw) -> Path:
    """Write the manifest and its diff sidecar; never overwrite. Returns the manifest path."""
    manifest, diff = build_manifest(root, spec, **kw)
    path = Path(root) / spec["freeze_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    side = path.parent / manifest["git_diff_file"]
    manifest["git_diff_file_sha256"] = sha256_bytes(diff.encode())
    with open(side, "x") as f:
        f.write(diff)
    with open(path, "x") as f:
        f.write(json.dumps(manifest, indent=1))
    return path


def manifest_problems(manifest, spec: dict, root: Path, *, current_analysis=True) -> list[str]:
    """Static checks of a manifest against the spec; with `current_analysis`, also that the
    analysis files and the protocol documents still have their frozen bytes."""
    if not isinstance(manifest, dict):
        return ["freeze manifest is not a JSON object"]
    out = []
    for k, want in (("manifest_version", MANIFEST_VERSION), ("experiment", spec["name"]),
                    ("run_dir", spec["run_dir"]), ("budget_s", spec["budget_s"]), ("commands", commands_for(spec)),
                    ("required_model", dict(spec["required_model"])), ("analysis_files", list(ANALYSIS_FILES))):
        if manifest.get(k) != want:
            out.append(f"freeze manifest {k} {manifest.get(k)!r} differs from the spec ({want!r})")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("base_git_head"))):
        out.append(f"freeze manifest base_git_head {manifest.get('base_git_head')!r}")
    if not finite(manifest.get("frozen_unix")):
        out.append(f"freeze manifest frozen_unix {manifest.get('frozen_unix')!r} missing or not finite")
    files = manifest.get("files_sha256") if isinstance(manifest.get("files_sha256"), dict) else {}
    for f in spec["executed_sources"] + ANALYSIS_FILES:
        if f not in files:
            out.append(f"freeze manifest does not hash {f}")
    out += [f"freeze manifest stack: {p}" for p in stack_problems(manifest.get("stack") or {}, spec)]
    docs = manifest.get("documents_sha256") if isinstance(manifest.get("documents_sha256"), dict) else {}
    if set(docs) != set(DOCUMENTS):
        out.append(f"freeze manifest documents {sorted(docs)}, expected {list(DOCUMENTS)}")
    if current_analysis:
        root = Path(root)
        for f in ANALYSIS_FILES:
            p = root / f
            if not p.is_file() or files.get(f) != sha256_file(p):
                out.append(f"{f} differs from its frozen bytes (analysis code changed after the freeze)")
        for d in DOCUMENTS:
            p = root / d
            if not p.is_file() or docs.get(d) != sha256_file(p):
                out.append(f"{d} differs from its frozen bytes")
    return out


def verify_tree(root: Path, manifest: dict, spec: dict, *, stack=None, environ=None) -> list[str]:
    """Before launching: the working tree, the documents, the stack and the launch environment
    still equal the freeze."""
    root = Path(root)
    out = manifest_problems(manifest, spec, root) + launch_env_problems(spec, environ)
    src = source_identity(root)
    if src["git_head"] != manifest.get("base_git_head"):
        out.append(f"HEAD {src['git_head']} != frozen base {manifest.get('base_git_head')}")
    if src["git_diff_head_sha256"] != manifest.get("git_diff_head_sha256"):
        out.append("git diff HEAD differs from the frozen diff")
    frozen = manifest.get("files_sha256") or {}
    if src["files_sha256"] != frozen:
        changed = sorted(f for f in set(frozen) | set(src["files_sha256"])
                         if frozen.get(f) != src["files_sha256"].get(f))
        out.append(f"source files differ from the freeze: {changed}")
    for p, h in (manifest.get("untracked_files_sha256") or {}).items():
        if not (root / p).is_file() or sha256_file(root / p) != h:
            out.append(f"untracked file {p} differs from the freeze")
    side = Path(root) / Path(spec["freeze_manifest"]).parent / str(manifest.get("git_diff_file"))
    if not side.is_file() or sha256_file(side) != manifest.get("git_diff_file_sha256"):
        out.append(f"diff sidecar {side.name} is missing or differs")
    stack = current_stack(root, spec) if stack is None else stack
    out += [f"current stack: {p}" for p in stack_problems(stack, spec)]
    for k in ("kernel", "cpu"):
        if stack.get(k) != (manifest.get("stack") or {}).get(k):
            out.append(f"current {k} {stack.get(k)!r} != frozen {(manifest.get('stack') or {}).get(k)!r}")
    return out


# ---- budgeted batch execution
STOP_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
WAIT_SIGNALS = STOP_SIGNALS | {signal.SIGCHLD}
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)  # loaded before any fork, as kvlab/perf_harness.py does
PR_SET_PDEATHSIG = 1


def _write_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # a member we may not signal: it is not ours to clean up
        return False


def _kill_group(pgid: int, grace_s: float, proc=None) -> dict:
    """SIGTERM, then SIGKILL after `grace_s`, to one process group this module created
    (`start_new_session=True`, so pgid == the session process's pid). Bounded: at most
    2 x `grace_s`. `proc` (the group leader) is reaped while waiting; an unreaped zombie would
    keep the group signalable. A process stuck in uninterruptible sleep can outlive SIGKILL;
    that is recorded as group_gone=False, not waited for."""
    out = {"pgid": pgid, "sigterm": False, "sigkill": False}

    def gone(deadline):
        while time.monotonic() < deadline:
            if proc is not None:
                proc.poll()
            if not _group_alive(pgid):
                return True
            time.sleep(0.02)
        return False
    if not _group_alive(pgid):
        out["group_gone"] = True
        return out
    try:
        os.killpg(pgid, signal.SIGTERM)
        out["sigterm"] = True
    except ProcessLookupError:  # the group ended in between
        pass
    if not gone(time.monotonic() + grace_s):
        try:
            os.killpg(pgid, signal.SIGKILL)
            out["sigkill"] = True
        except ProcessLookupError:
            pass
        gone(time.monotonic() + grace_s)
    if proc is not None:
        proc.poll()
    out["group_gone"] = not _group_alive(pgid)
    return out


def _child_setup(wrapper_pid: int) -> None:
    """Runs in the session child between fork and exec (single-threaded wrapper).
    - Unblock the stop signals: the wrapper blocks them, and a signal mask survives exec.
    - SIGTERM when the wrapper dies, even by SIGKILL (PR_SET_PDEATHSIG, as the runner's GPU
      monitor uses).
    - If the wrapper already died before that registration, exit at once instead of running
      unsupervised."""
    signal.pthread_sigmask(signal.SIG_SETMASK, ())
    _LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != wrapper_pid:
        os._exit(75)


def _pending_stop_signal():
    """A stop signal that arrived while blocked, or None (non-blocking)."""
    info = signal.sigtimedwait(STOP_SIGNALS, 0)
    return None if info is None else signal.Signals(info.si_signo).name


def _wait(proc, deadline: float, clock):
    """Wait for the session to exit, a stop signal, or the deadline, whichever is first.
    Signals are received synchronously (sigtimedwait on blocked signals, SIGCHLD included),
    so the exit is observed when it happens, not at the next poll. Returns (stop or None,
    observation time). An exit first observed at or after the deadline is a deadline overrun,
    never a success."""
    while True:
        if proc.poll() is not None:
            now = clock()
            if now >= deadline:
                return {"reason": "deadline", "detail": "the session's exit was first observed after the deadline"}, now
            return None, now
        now = clock()
        if now >= deadline:
            return {"reason": "deadline", "detail": "the session was still running at the deadline"}, now
        info = signal.sigtimedwait(WAIT_SIGNALS, deadline - now)
        if info is not None and info.si_signo in STOP_SIGNALS:
            return {"reason": "operator_signal", "detail": f"{signal.Signals(info.si_signo).name} received"}, clock()
        # SIGCHLD or timeout: re-poll and recheck the deadline


def _runner_summary(p: Path):
    """What the runner itself recorded (meta and failures), for the operator. Not validated:
    only scripts/m3_analyze.py judges the evidence."""
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(r, dict):
        return None
    m = r.get("meta") if isinstance(r.get("meta"), dict) else {}
    f = r.get("failures")
    return {"correctness_all": m.get("correctness_all"), "n_block_failures": len(f) if isinstance(f, list) else None,
            "n_excluded": m.get("n_excluded"), "n_measured": m.get("n_measured"), "n_samples": m.get("n_samples")}


def run_batch(root: Path, manifest: dict, manifest_sha256: str, *, budget_s=None, grace_s: float = 15.0,
              verification=None, log=print, clock=time.monotonic) -> dict:
    """Run the manifest's commands one after another, each in its own new process group, within
    one deadline: `budget_s` of `clock()` time from launching the first session process.

    Stopping rule (no retries, replacements or extra sessions). Nothing further is launched
    after any of:
    - a nonzero session exit (any code; the session's report, log and monitor CSV are kept);
    - a session that exited 0 without a parseable session file;
    - the deadline: reached while a session runs, a session exit first observed at or after it,
      or reached before the next launch;
    - a stop signal (SIGINT, SIGTERM, SIGHUP);
    - an error in the wrapper itself (re-raised after cleanup).

    The execution status is "succeeded" only if every planned session was launched and exited 0
    by itself, before the deadline, leaving a parseable session file. Otherwise it is "failed",
    and `stop` says why. This is the execution status only: whether the evidence is complete and
    valid is decided by the analysis (kvlab/m3_validate.py), never by this record.

    Signals: the stop signals and SIGCHLD are blocked for the whole call and received only
    synchronously, in `_wait()` and before each launch. No asynchronous exception can therefore
    land inside `Popen`, between launch and the wait, or inside the cleanup. The cleanup
    (terminate and reap the active session's group, bounded; remove only that session's scratch;
    finalize the record) always runs, and later signals are only recorded. The previous signal
    mask is restored on return.

    Returns the batch record, which is also written to <run_dir>/batch_record.json after every
    state change."""
    root = Path(root)
    run_dir = root / manifest["run_dir"]
    budget_s = manifest["budget_s"] if budget_s is None else budget_s
    cmds = list(manifest["commands"])
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, WAIT_SIGNALS)
    try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        rec_path = run_dir / BATCH_RECORD
        rec = {"record_version": BATCH_RECORD_VERSION, "experiment": manifest["experiment"],
               "freeze_manifest_sha256": manifest_sha256, "pre_launch_verification": verification,
               "budget_s": budget_s, "clock": "time.monotonic; deadline = first launch + budget_s",
               "wrapper_pid": os.getpid(), "execution_status": "running", "stop": None,
               "launched_at": None, "launched_unix": None, "sessions": [], "not_launched": [],
               "elapsed_s": None, "finalized_s": None, "signals_received": []}
        _write_atomic(rec_path, rec)
        state = {"t0": None, "deadline": None, "active": None, "stop": None, "error": None, "grace_s": grace_s}
        try:
            _launch_all(root, run_dir, rec, rec_path, cmds, budget_s, state, log, clock)
        except Exception as e:  # noqa: BLE001  (recorded, cleaned up, then re-raised)
            state["error"] = e
            state["stop"] = {"reason": "wrapper_error", "session": rec["sessions"][-1]["session"]
                             if rec["sessions"] else None, "detail": repr(e)[:500]}
        finally:
            _finish(root, run_dir, rec, rec_path, cmds, state, grace_s, clock)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    if state["error"] is not None:
        raise state["error"]
    return rec


def _launch_all(root, run_dir, rec, rec_path, cmds, budget_s, state, log, clock):
    for cmd in cmds:
        sid = cmd["session"]
        sig = _pending_stop_signal()
        if sig:
            state["stop"] = {"reason": "operator_signal", "session": None, "detail": f"{sig} before launching {sid}"}
            return
        if state["t0"] is not None and clock() >= state["deadline"]:
            state["stop"] = {"reason": "deadline", "session": None, "detail": f"deadline reached before launching {sid}"}
            return
        entry = {"session": sid, "argv": cmd["argv"], "env": cmd["env"], "log": f"logs/session_{sid}.log",
                 "pid": None, "started_s": None, "ended_s": None, "returncode": None, "killed": None,
                 "leftover_cleanup": None, "session_json": None, "monitor_csv_present": None,
                 "runner_reported": None, "scratch_removed": None}
        rec["sessions"].append(entry)
        with open(run_dir / entry["log"], "wb") as logf:
            if state["t0"] is None:
                state["t0"] = clock()
                state["deadline"] = state["t0"] + budget_s
                rec["launched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                rec["launched_unix"] = time.time()
            proc = subprocess.Popen(cmd["argv"], cwd=root, env={**os.environ, **cmd["env"]}, stdout=logf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                    preexec_fn=lambda pid=os.getpid(): _child_setup(pid))
            state["active"] = (proc, entry)  # owned from here until reaped, whatever happens
        entry.update(pid=proc.pid, started_s=clock() - state["t0"])
        _write_atomic(rec_path, rec)
        log(f"launched {sid} (pid {proc.pid}) at {entry['started_s']:.1f} s")
        stop, now = _wait(proc, state["deadline"], clock)
        if proc.poll() is not None:  # exited by itself (reaped by poll), possibly too late
            state["active"] = None
            entry.update(ended_s=now - state["t0"], returncode=proc.returncode, killed=False)
            _after_session(root, run_dir, entry, sid, state["grace_s"])
        if stop:  # deadline or signal; a session still running is terminated and reaped by _finish()
            state["stop"] = {**stop, "session": sid,
                             "detail": f"{stop['detail']} (observed {now - state['t0']:.3f} s after the first launch)"}
            state["stop_s"] = now - state["t0"]
            return
        if proc.returncode != 0:
            state["stop"] = {"reason": "nonzero_exit", "session": sid, "returncode": proc.returncode,
                             "detail": f"exit code {proc.returncode}; runner-recorded summary (not validated): "
                                       f"{entry['runner_reported']}"}
            return
        if not entry["session_json"]["parses"]:
            state["stop"] = {"reason": "missing_report", "session": sid,
                             "detail": f"exit 0 without a parseable session file: {entry['session_json']}"}
            return
        _write_atomic(rec_path, rec)
        log(f"finished {sid}: exit 0 at {entry['ended_s']:.1f} s")


def _after_session(root, run_dir, entry, sid, grace_s):
    """After a session ended (by itself or killed): stop what it left in its own group, record
    its files, and remove only its own disk-tier scratch directory if it is still there."""
    if entry.get("pid") and _group_alive(entry["pid"]):
        entry["leftover_cleanup"] = _kill_group(entry["pid"], grace_s)
    sj = run_dir / f"session_{sid}.json"
    entry["session_json"] = _file_state(sj)
    entry["runner_reported"] = _runner_summary(sj) if entry["session_json"]["present"] else None
    entry["monitor_csv_present"] = (run_dir / f"session_{sid}.gpu_monitor.csv").is_file()
    scratch = scratch_dir(root, run_dir.name, sid)
    if scratch.is_dir():
        shutil.rmtree(scratch)
        entry["scratch_removed"] = str(scratch.relative_to(root))


def _finish(root, run_dir, rec, rec_path, cmds, state, grace_s, clock):
    """Always runs. Signals are still blocked, so nothing can interrupt it; signals that arrive
    meanwhile are recorded afterwards."""
    active = state.get("active")
    if active is not None:  # launched and not yet reaped: terminate its group and reap it (bounded)
        proc, entry = active
        entry["killed"] = _kill_group(proc.pid, grace_s, proc)
        entry.update(ended_s=None if state["t0"] is None else clock() - state["t0"], returncode=proc.poll())
        _after_session(root, run_dir, entry, entry["session"], grace_s)
        state["active"] = None
    t0, stop = state["t0"], state["stop"]
    launched = {e["session"] for e in rec["sessions"]}
    rec["not_launched"] = [c["session"] for c in cmds if c["session"] not in launched]
    rec["stop"] = stop
    ends = [e["ended_s"] for e in rec["sessions"] if isinstance(e.get("ended_s"), (int, float))]
    # the budget limits execution: first launch to the last observed exit (or the stop decision)
    rec["elapsed_s"] = None if t0 is None else max(ends + [state.get("stop_s", 0.0)], default=None)
    rec["finalized_s"] = None if t0 is None else clock() - t0
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec["execution_status"] = "succeeded" if stop is None and not rec["not_launched"] else "failed"
    while True:  # drain signals received while blocked, so none fires after the mask is restored
        info = signal.sigtimedwait(WAIT_SIGNALS, 0)
        if info is None:
            break
        if info.si_signo in STOP_SIGNALS:
            rec["signals_received"].append(signal.Signals(info.si_signo).name)
    _write_atomic(rec_path, rec)


def exit_code(rec) -> int:
    """The wrapper's exit status: 0 only if the execution succeeded, else 3."""
    return 0 if rec.get("execution_status") == "succeeded" else 3


def _file_state(p: Path) -> dict:
    if not p.is_file():
        return {"present": False, "parses": False, "sha256": None, "bytes": None}
    data = p.read_bytes()
    try:
        ok = isinstance(json.loads(data), dict)
    except ValueError:
        ok = False
    return {"present": True, "parses": ok, "sha256": sha256_bytes(data), "bytes": len(data)}


def batch_problems(rec, manifest: dict, manifest_sha256: str, run_dir: Path, spec: dict) -> list[str]:
    """Evidence-relevant problems in the batch record. A nonzero session exit is an execution
    failure, but not by itself an evidence problem: the analysis judges the session files
    against the spec. What does make the evidence invalid here:
    - the record cannot be trusted or bound: version, experiment, manifest SHA-256, budget,
      pre-launch verification, launch before the freeze, or a record never finalized;
    - a stop other than a nonzero exit: deadline, signal, missing report or wrapper error,
      i.e. a session timed out, was killed or did not report;
    - planned sessions never launched (their evidence is missing);
    - a launch after a nonzero exit (stopping rule violated), a killed session, a session file
      that differs from the one recorded, or a session ending at or after the deadline;
    - a contradictory execution record: status, stop, launched sequence, exits, session-file
      states and timing that no run of run_batch() can produce (execution_consistency_problems,
      checked in both directions)."""
    if not isinstance(rec, dict):
        return ["batch record is not a JSON object"]
    out = []
    for k, want in (("record_version", BATCH_RECORD_VERSION), ("experiment", spec["name"]),
                    ("freeze_manifest_sha256", manifest_sha256), ("budget_s", spec["budget_s"])):
        if rec.get(k) != want:
            out.append(f"batch record {k} {rec.get(k)!r}, expected {want!r}")
    v = rec.get("pre_launch_verification")
    if not isinstance(v, dict) or v.get("problems") != []:
        out.append(f"batch record has no passing pre-launch verification: {v!r}")
    fr = manifest.get("frozen_unix")
    if not finite(rec.get("launched_unix")) or not finite(fr) or rec["launched_unix"] < fr:
        out.append(f"batch launched at {rec.get('launched_unix')!r}, not after the freeze ({fr!r})")
    status, stop = rec.get("execution_status"), rec.get("stop")
    if status not in ("succeeded", "failed"):
        out.append(f"batch record not finalized (execution_status {status!r}): the wrapper did not finish, so "
                   f"the session exits and the deadline cannot be verified")
    if stop is not None and (not isinstance(stop, dict) or stop.get("reason") != "nonzero_exit"):
        out.append(f"batch execution stopped by {stop!r}: a session timed out, was interrupted or did not report")
    entries = rec.get("sessions") if isinstance(rec.get("sessions"), list) else []
    cmds = manifest.get("commands") or []
    planned = [c["session"] for c in cmds]
    names = [e.get("session") if isinstance(e, dict) else None for e in entries]
    if names != planned[:len(names)] or rec.get("not_launched") != planned[len(names):]:
        out.append(f"batch sessions {names}, not launched {rec.get('not_launched')!r}: not the planned order {planned}")
    if rec.get("not_launched"):
        out.append(f"planned sessions never launched, so their evidence is missing: {rec.get('not_launched')!r}")
    budget = spec["budget_s"]
    for i, (e, c) in enumerate(zip(entries, cmds)):
        if not isinstance(e, dict):
            continue
        sid, rc = c["session"], e.get("returncode")
        if e.get("argv") != c["argv"] or e.get("env") != c["env"]:
            out.append(f"batch {sid}: launched command {e.get('argv')!r} {e.get('env')!r} != frozen {c['argv']!r} {c['env']!r}")
        if not isinstance(rc, int) or e.get("killed") is not False:
            out.append(f"batch {sid}: exit {rc!r}, killed {e.get('killed')!r}: the session did not exit by itself")
        elif rc != 0 and i != len(entries) - 1:
            out.append(f"batch {sid}: exit {rc}, yet a later session was launched (stopping rule violated)")
        # the recorded file state (presence, parseability, digest, size) must be the actual one
        actual = _file_state(run_dir / f"session_{sid}.json")
        if e.get("session_json") != actual:
            out.append(f"batch {sid}: session file differs from the one the batch recorded "
                       f"(recorded {e.get('session_json')!r}, actual {actual!r})")
        if not actual["present"]:
            out.append(f"batch {sid}: no session file")
        if not finite(e.get("ended_s")) or e["ended_s"] >= budget:
            out.append(f"batch {sid}: ended at {e.get('ended_s')!r} s, not a finite time before the {budget} s deadline")
    el = rec.get("elapsed_s")
    if not finite(el) or el >= budget:
        out.append(f"batch elapsed {el!r} s is missing, not finite or not below the {budget} s budget")
    out += execution_consistency_problems(rec, [e for e in entries if isinstance(e, dict)])
    return out


def _exit_code(e: dict):
    """A recorded exit code, or None if missing or not an int (JSON true/false is not one)."""
    rc = e.get("returncode")
    return rc if type(rc) is int else None


def execution_consistency_problems(rec: dict, entries: list) -> list[str]:
    """Execution status, stop, launched sequence, exits, reports and timing must describe one
    run that run_batch() can produce, checked in both directions. A contradictory record is
    rejected before any performance analysis. Only two execution shapes are acceptable
    evidence; every other stop is reported as an evidence problem by batch_problems():
    - succeeded: no stop, nothing unlaunched, every launched session exited 0 by itself with a
      parseable session file;
    - nonzero_exit: status failed; the stop names the last launched session and its actual
      nonzero exit code; that session exited by itself; every earlier one exited 0 with a
      parseable session file. Nothing after it was launched (checked with the planned order).
    In both, elapsed_s is the last session's end, and the sessions do not overlap in time."""
    out = []
    status, stop = rec.get("execution_status"), rec.get("stop")
    reason = stop.get("reason") if isinstance(stop, dict) else None
    exits = [(e.get("session"), e.get("returncode"), e.get("killed")) for e in entries]
    self_nonzero = [e.get("session") for e in entries if e.get("killed") is False and _exit_code(e) not in (None, 0)]
    # status <-> stop and launch sequence
    if status == "succeeded" and (stop is not None or rec.get("not_launched")):
        out.append(f"batch execution_status 'succeeded', yet stop {stop!r} and not launched {rec.get('not_launched')!r}")
    if status == "failed" and stop is None:
        out.append("batch execution_status 'failed' without a stop reason")
    if stop is None and rec.get("not_launched"):
        out.append(f"batch has no stop, yet sessions were never launched: {rec.get('not_launched')!r}")
    # a nonzero-exit stop <-> the actual exit of the last launched session (P1: stop vs exits)
    if reason == "nonzero_exit":
        last = entries[-1] if entries else {}
        actual = (last.get("session"), _exit_code(last))
        if last.get("killed") is not False or actual[1] in (None, 0):
            out.append(f"batch stop records a nonzero exit of {stop.get('session')!r} ({stop.get('returncode')!r}), "
                       f"but the last launched session {actual[0]!r} exited {last.get('returncode')!r} "
                       f"(killed {last.get('killed')!r}); recorded exits {exits}")
        elif (stop.get("session"), stop.get("returncode")) != actual or type(stop.get("returncode")) is not int:
            out.append(f"batch stop names {stop.get('session')!r} ({stop.get('returncode')!r}), "
                       f"but the last launched session exited {actual}")
        if status != "failed":
            out.append(f"batch execution_status {status!r} with a nonzero-exit stop")
    # an actual nonzero exit <-> a recorded stop and a failed status
    if self_nonzero and (stop is None or status != "failed"):
        out.append(f"batch sessions {self_nonzero} exited nonzero, yet stop {stop!r} and execution_status {status!r}")
    if reason == "nonzero_exit" or stop is None:  # the two shapes accepted as evidence
        for i, e in enumerate(entries):
            if _exit_code(e) is None:
                out.append(f"batch {e.get('session')}: exit code {e.get('returncode')!r} is not an integer")
            stopped_here = reason == "nonzero_exit" and i == len(entries) - 1
            if not stopped_here:
                sj = e.get("session_json")
                if _exit_code(e) != 0 or not (isinstance(sj, dict) and sj.get("parses") is True):
                    out.append(f"batch {e.get('session')}: exit {e.get('returncode')!r}, session file {sj!r}: "
                               f"the wrapper would have stopped here")
        # timing: sequential sessions; elapsed_s is the last session's end
        times = [(e.get("started_s"), e.get("ended_s")) for e in entries]
        flat = [x for pair in times for x in pair]
        if not all(finite(x) for x in flat) or any(x < 0 for x in flat) or \
                any(b < a for a, b in zip(flat, flat[1:])):
            out.append(f"batch session times {times} are missing, not finite, negative or overlapping")
        elif entries and rec.get("elapsed_s") != flat[-1]:
            out.append(f"batch elapsed_s {rec.get('elapsed_s')!r} is not the last session's end ({flat[-1]})")
    return out


def main_freeze(spec, root) -> int:
    try:
        path = freeze(root, spec)
    except Refused as e:
        print("freeze REFUSED:\n" + "\n".join(f"- {p}" for p in e.problems))
        return 1
    print(f"wrote {path} (sha256 {sha256_file(path)})\nRecord this SHA-256 outside the repository before measuring.")
    return 0


def load_manifest(root, spec, path=None):
    p = Path(root) / spec["freeze_manifest"] if path is None else Path(path)
    data = p.read_bytes()
    return json.loads(data), sha256_bytes(data), p


def _load_or_report(root, spec, path):
    try:
        return load_manifest(root, spec, path)
    except (OSError, ValueError) as e:
        print(f"REFUSED: freeze manifest missing or unreadable: {e}")
        return None


def main_verify(spec, root, path=None) -> int:
    loaded = _load_or_report(root, spec, path)
    if loaded is None:
        return 1
    manifest, sha, p = loaded
    problems = verify_tree(root, manifest, spec)
    print(f"freeze manifest {p} sha256 {sha}")
    print("verification: " + ("PASS" if not problems else "FAIL\n" + "\n".join(f"- {x}" for x in problems)))
    return 0 if not problems else 1


def main_run(spec, root, path=None, expect_sha=None) -> int:
    loaded = _load_or_report(root, spec, path)
    if loaded is None:
        return 1
    manifest, sha, p = loaded
    problems = verify_tree(root, manifest, spec)
    if expect_sha and sha != expect_sha:
        problems.append(f"freeze manifest sha256 {sha} != expected {expect_sha}")
    if (Path(root) / spec["run_dir"]).exists():
        problems.append(f"{spec['run_dir']} already exists (no reruns or replacement sessions)")
    if problems:
        print("launch REFUSED:\n" + "\n".join(f"- {x}" for x in problems))
        return 1
    ver = {"problems": [], "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "manifest": str(p)}
    try:
        rec = run_batch(root, manifest, sha, verification=ver, log=lambda m: print(m, flush=True))
    except Exception:  # noqa: BLE001  the record is finalized and the session cleaned up; show the error
        traceback.print_exc()
        print("batch execution FAILED: wrapper error (see the traceback and batch_record.json)", flush=True)
        return 3
    print(json.dumps({k: rec[k] for k in ("execution_status", "stop", "not_launched", "elapsed_s",
                                          "signals_received")}, indent=1))
    for e in rec["sessions"]:
        if e["returncode"] != 0:
            print(f"session {e['session']}: exit {e['returncode']}, killed {bool(e['killed'])}; runner-recorded "
                  f"summary (not validated): {e['runner_reported']}")
    print(f"batch execution {rec['execution_status'].upper()}; run scripts/m3_analyze.py to classify the evidence "
          f"(it reports recorded correctness failures in every case)", flush=True)
    return exit_code(rec)


if __name__ == "__main__":
    sys.exit("use scripts/m3_confirm.py")
