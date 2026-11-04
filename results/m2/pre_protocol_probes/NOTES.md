# Pre-protocol probes (exploratory; 2026-11-03 13:00–13:12; not evidence for any reported number)

These scratch scripts ran once each, before `docs/M2_PROTOCOL.md` was fixed. They
explain protocol choices. Their console output was not saved to files. The figures
below are copied from it and are approximate. All used the project `.venv` and the
cached model, and wrote only under `.kvcache/m2/probe*` (removed afterwards).

| Script | Question | Observed |
|---|---|---|
| probe_full.py | Why is full N=257 ~82 ms in the first smoke runs? | Isolated prefill N=257 took 25.6 ms, launch-bound (~1,100 kernel launches, ~22 ms of GPU time). Medians are noisy across blocks. |
| probe_gap.py | Does idle time before a sample matter? | Back-to-back bursts are stable (full N=1088 at 63.8–65.2 ms). With idle gaps, medians rise (full N=257: 27 → 41 ms at 0.5 s) and spikes reach 233 ms. |
| probe_harness.py | Does GPU pre-conditioning or `gc.collect` change harness medians? | No material change. Full L=256 S=1 at 26.8 ms; occasional +50 ms spikes on restore paths. `gc.collect()` costs ~148 ms. |
| probe_corr.py | Does the correctness check slow later samples? | No (26.8 → 26.9 ms); 0 device allocations per sample. |
| probe_smi.py | Does an `nvidia-smi` query slow later samples? | No lasting effect. |
| probe_flow.py | Replay of the measurement flow | Full N=1088 switches between ~64 and ~80 ms regimes lasting seconds. |
| probe_flow_pinned.py | Same, process pinned to one vCPU | The regimes persist (78–92 ms), so CPU placement is not the cause. |
| probe_clk.py | Latency vs GPU clock under continuous load | Clock 2325–2370 MHz; full N=1088 stable at 64.1 ms. |
| probe_regime.py | Allocator and clock in the slow regime | Slow regime = SM clock 2280 → ~1750–1815 MHz; allocator unchanged; power 49–146 W. |

Consequences for the protocol (see §4 there): interleaved paths with paired
winners, a concurrent read-only clock monitor, GC once per repetition, and
reporting with a platform-drift caveat. No power or driver setting was changed.
