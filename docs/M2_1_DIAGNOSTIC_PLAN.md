# M2.1 diagnostic plan (written 2026-11-03, before any M2.1 GPU sample)

Scope: narrow down the restore-path "slow mode" seen in M2. Finding its root
cause, or a speedup, is not required. No runtime policy, kernel, dependency,
power or driver change is made.
- One GPU process at a time.
- About 45 minutes of extra GPU time at most.
- The full M2 grid is not rerun.

The CPU-only reanalysis of the saved M2 data follows its own rules, fixed
in `scripts/m2_1_reanalyze.py` before that script first ran.

## What the saved M2 data already shows (CPU reanalysis, not new measurement)
In the M2 instrumented samples, slow samples are slower in **every phase**
that the host thread drives or waits on:

| Phase | Normal | Slow |
|---|---|---|
| host restore loop (L = 4096) | 57 ms | 146 ms |
| D2H store, per chunk | 4.4 ms | 8 ms |
| select | 0.1 ms | 0.2 ms |
| cached forward | ~17–25 ms | ~60–70 ms (a floor) |

Two things do not change in slow samples: the restore phase of
`disk_evicted` (dominated by I/O wait), and the GPU-bound full prefill at
L ≥ 1024. This suggests a slowdown in host-side submission or
synchronization, not in GPU kernel execution. It is a hypothesis, to be
tested here.

## Hypotheses
- **H1 — GPU clock.** Slow events are explained by a lower SM clock.
- **H2 — GPU execution.** In slow events, GPU kernels themselves take longer.
- **H3 — host-side submission or synchronization.** In slow events, the host
  thread takes longer to submit the forward (kernel launches), and to issue
  and complete synchronous copies. The GPU waits for work.
  - H3a: the thread is on-CPU but slower; thread CPU time grows with wall
    time.
  - H3b: the thread is off-CPU or blocked in the driver; wall time grows,
    thread CPU time does not.
- **H4 — instrumentation.** The M2 synchronizing phase marks, or the GPU
  monitor, cause or amplify the slow mode.
- **H5 — path-specific trigger vs shared state.** The slow mode is either
  specific to restore work, or a process- or platform-wide state that also
  affects recompute but is hidden when recompute is GPU-bound.
- **H6 — allocator.** Slow samples coincide with device allocations
  (`cudaMalloc`) or allocator retries.
- **H7 — session.** The slow regime is a property of a process session, or of
  time within a session.

## Experiment (`scripts/m2_1_diagnose.py`, `kvlab/diag_harness.py`)
- **Cases:** (L, S) = (512, 1), (1024, 64), (4096, 64).
- **Paths:**
  - `full`: the M2 baseline, renamed in reports as "recompute from the
    prepared cache state" (prefix chunks are in store A; restore is not
    attempted; admission is unchanged);
  - `host`;
  - `disk` (warm page cache, as in M2);
  - `full_nocache` (diagnostic only): the same token list goes to a device
    tensor, followed by `kvlab.model.prefill(model, ids, None, 0)` and
    `int(argmax)`. There are no chain keys, no lookup and no admission. Its
    timing boundary is the same as the other paths. For S ∈ {1, 64} no
    complete new chunk exists, so `full` admits nothing either. The two
    should then differ only by chain keys and lookups (under 1 ms).
- **Modes**, interleaved within every repetition:
  - `plain`: the unchanged M2 `Bench.measure` (uninstrumented);
  - `events`: no device synchronization inside the region. At each phase
    boundary, the harness records a pre-created CUDA event, the wall clock
    and the thread CPU time (`time.thread_time_ns`);
  - `synced`: the unchanged M2 instrumented `Bench.measure`.
- **Sessions:** 4 fresh-process sessions with the M2 configuration (GPU
  monitor on).
  - Each session visits every case in two separate blocks, in a random
    permutation repeated twice.
  - Each block runs 3 warmup repetitions and then 12 measured repetitions.
  - In every repetition, `gc.collect()` runs, then all 12 (path, mode) pairs
    run in a fresh random order.
- **Per sample:**
  - session, block, repetition, order position;
  - `t_wall` and the monotonic start;
  - the validation fields of M2 (tier, restore length, admissions,
    residency);
  - thread `getrusage` deltas (context switches, CPU time, minor faults),
    taken just outside the timed region;
  - allocator counters (`num_device_alloc`, `num_device_free`,
    `num_alloc_retries`) before and after the sample.
- **Correctness, untimed, once per block:**
  - the M2 strict check (restored bytes equal; host == disk == resident,
    bitwise);
  - `full_nocache` logits bitwise equal to `full` logits.
- **Spike-probe audit:** the M2 probe's cached-forward function runs 20
  times. The check asserts that the cache holds 256 tokens before the forward
  and 257 after, and that the source K/V tensors are unchanged (pointer,
  shape, checksum).
- **Host sync inside the forward:** `torch.cuda.set_sync_debug_mode("warn")`
  runs around one cached forward, outside the timing loop. This tells
  whether "submission" time can include a wait.

**Optional, decided only by the outcome of the four sessions** (at most two
extra sessions, one factor each, labeled as diagnostic configurations):
- **F1:** the GPU monitor off (tests H4 for the monitor).
- **F2:** if H3b is supported, set the CUDA device schedule flag to spin
  (`cudaSetDeviceFlags(cudaDeviceScheduleSpin)` before context creation),
  process-local.

**Profiler, if available without installation:** one short session with
`torch.profiler` (CPU + CUDA). It runs `host` and `full_nocache` at
L = 512, S = 1, interleaved, about 40 repetitions. Each sample gets a
`record_function` range. Per sample, it extracts:
- the CPU-side CUDA runtime API time (`cudaLaunchKernel`, memcpy, sync
  calls);
- the GPU kernel time sum;
- the wall time.

The perturbation is recorded as profiled vs plain medians. Profiler numbers
are never used as latency estimates. No large trace is saved.

## Contrasts (slow = e2e > 1.3× the case/path/mode minimum of the session set, as in M2)
| Contrast | Distinguishes |
|---|---|
| Slow vs fast samples within the same reported SM-clock bin (monitor join) | H1 vs not-H1 |
| `events`: GPU-stream time from the restore mark to the forward's last kernel vs host wall submission time of the forward; profiler kernel-time sum, slow vs fast | H2 vs H3 |
| `events`: thread CPU time vs wall time during submission and the restore loop; involuntary/voluntary context switches | H3a vs H3b |
| Slow share in `plain` vs `events` vs `synced` (same repetitions); session without monitor (F1) | H4 |
| `full`/`full_nocache` forward **submission** time (events) in repetitions where restore is slow vs fast, including GPU-bound L = 4096 | H5 |
| Allocator counter deltas, slow vs fast | H6 |
| Slow share by session and block | H7 |

**Stopping rule:** stop adding experiments when the contrasts no longer
separate the remaining hypotheses, or when the budget is used.
- No slow sample is discarded.
- No path is delayed artificially.
- Root cause is claimed only if a contrast establishes it.
- Correlation with WSL, Windows scheduling, power sharing or a backend is
  reported as correlation.

## Amendment 1 (15:12, after the four baseline sessions, before any further GPU sample)
**What the baseline sessions showed** (`results/m2_1/diag_20261103T150340/diag_analysis.md`):
- **H1 is excluded:** slow and fast samples have the same median SM clock.
- **H4 (synchronizing marks) is excluded:** plain, events and synced have
  similar slow shares.
- **H6 is excluded:** there are no device allocations.
- **H7 (session-level) is excluded.**
- **H3a, not H3b:** in slow samples, the host thread is on-CPU (thread CPU
  ≈ wall, runqueue wait ≈ 0) for 2.5–3.7× longer while it submits the
  forward and runs the synchronous copy loop.
- **H5, shared state:** in the same repetitions, the recompute path's forward
  submission slows too (17 → 35–38 ms). This is hidden in its e2e by GPU
  execution time.

**The open question** is which of these holds:
- **H8a:** the host thread executes *everything* slower (vCPU speed or
  hypervisor time, which is invisible to the guest; WSL reports no steal
  time);
- **H8b:** only CUDA driver/runtime calls (launches, synchronous copies)
  become slower.

The saved data already leans to H8b: the CPU-only phases (SHA-256 chain
keys, CRC32 in the evicted read) do not grow. A direct test follows.

**Probe session** (2 sessions, baseline configuration). Four untimed-in-M2
"probe" items join the per-repetition shuffle. Each is timed like a sample
(wall and thread CPU):
- `probe_cpu_sha`: SHA-256 of a fixed 8 MiB buffer (pure CPU, C code);
- `probe_cpu_py`: a fixed pure-Python loop;
- `probe_launch`: 500 tiny in-place kernels on a 1-element device tensor,
  then one sync (launch path);
- `probe_copy`: 56 synchronous H2D copies of 128 KiB from a pinned buffer
  (the restore's copy pattern).

The contrast compares probe medians in repetitions where the restore samples
are slow vs fast:
- H8a predicts that every probe slows;
- H8b predicts that only `probe_launch` and `probe_copy` slow.

**F1 (planned option, run):** one session with the GPU monitor off. The
monitor polls the driver through NVML every 200 ms, so it is a candidate
source of driver contention.

**F2 (spin schedule): not run.** It was conditional on H3b, which the data
does not support.

**Profiler session: run as planned.**

## Amendment 2 (15:27, after the probe, F1 and profiler sessions, before the last session)
**Results so far** (`diag_probes_*`, `diag_nomonitor_*`, `diag_profiler_*`):
- **H8b supported, H8a rejected.** In restore-slow repetitions, the pure-CPU
  probes are unchanged (ratio 0.99–1.01), while `probe_launch` is 4.0× and
  `probe_copy` 2.2× slower, with thread CPU equal to wall time.
- **Profiler.** GPU kernel time is unchanged in slow samples (9.92 vs
  9.93 ms), while the CPU-side `cudaLaunchKernel` time rises from 7.7 to
  23.3 µs per call.
- **F1 (monitor off).** The slow mode persists.

**Last contrast (one session; then stop).** Two probes, `probe_syscall_getppid`
(20,000 `getppid` calls) and `probe_syscall_read` (5,000 reads of 4 KiB from
`/dev/zero`), test whether *every* guest system call slows in the slow
regime, or only the CUDA driver/runtime path. All other settings are those
of the probe sessions.
