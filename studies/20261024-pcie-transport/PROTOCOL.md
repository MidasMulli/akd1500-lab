# Transport follow-up to PCIe fabric characterization

Chosen prospectively after all three primary resident CNN widths passed exact
scores and full learning-state checks. Primary source/fixtures/results remain
unchanged. This is a separate characterization branch, not a silent rerun.

Reuse the exact RGB32/C128 image, feature, program and initial-learning-state
fixtures. Explicit profiles: 0 mapped MMIO, 1 kernel DMA reads >=128 bytes,
2 kernel DMA reads+writes >=128 bytes, 3 kernel DMA writes >=128 bytes.
The installed driver and its configuration remain unchanged. Physical work
uses existing bounded pread/pwrite support, no register sweep or driver update.

First require byte-exact zero-forward copy checks in both directions at
128/512/3072 bytes. Then eight-image serial learning smokes separately for
profiles 1/2/3, no fallback. Inspect before 96-image queued cases. Compare
learning and frozen controls, batch eight with nine descriptor slots. First
cycle 0/1/2/3; repeat cycles 3/2/1/0 and 1/3/0/2, alternate learning/frozen order.
Every score and full learner RAM at each eight-image boundary must match the
independent CPU reference. All arrays and transfer-call counts retained.

32 physical sessions, 3500-forward cap, individual 120-second timeout. Same
2026-10-24 16:00 UTC scientific cutoff and 16:05 cleanup deadline. Stop this
branch on any output/state/copy failure. Timing excludes program setup and RAM
diagnostics. The source keeps all math, native programs and task state fixed.
Temporal repeats on one initialization/chip are not independent scientific
seeds. CPU baseline is the primary study's two verified full-pipeline NEON
controls; no claim of globally optimal CPU or a calibrated energy comparison.
