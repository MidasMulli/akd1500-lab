# Native primitives: physical execution, scaling and CPU implementation

New direct user approval after distinguishing native hardware capability from
CPU emulation and measured advantage. Prior studies remain closed. No game or
small association task substitutes for this characterization.

## A. Matched fixed computation

Binary input512, positive binary weights32/output, threshold15, two group counts.
Seed104101 weights have identical prefixes; FNP and CNP graphs use exactly the
same mathematical weights, thresholds, input bytes and outputs at each width.
FNP128/256/512 and CNP128/256/512/1024/2048. Map every cell prospectively with
ordinary compiler output, single pass, no FNP2, no custom learning adapter.
No guarantee that compiler-requested partitioning succeeds. Preserve failures;
do not swap representations or mapping policies to rescue a failed cell.

Input bank seed104102:160 events,32 per density0/8/64/256/512 in interleaved order.
Smoke selects the first8 from each density (40 events), then the unchanged full
bank. Every measured fixed-memory region must match compiler packets before and
after; every output independently equals binary overlap/threshold/group sums.
Reuse metal_bench instrumentation as primitive_bench, only widening the allowed
fixed-region count8→32 to capture larger ordinary compiler graphs.

Stage A1: FNP128 smoke then CNP128 smoke, inspect before full128 pair.
Stage A2: full256 pair, inspect before512 pair. Only after exact512 pairs may
fullCNP1024 then2048 run. Compiler-rejected cells remain closed. Then two temporal
repetitions per qualified cell, reverse cell order on the first repetition.
These are temporal repeats of one weight/input bank, not independent tasks.
Each actor adds48 frozen forwards for three matched serial/queued8-input suffixes.
DMA monitor cycles are buffer service, not arithmetic-only NPU cycles. Report
serial hot-path wall/caller CPU separately and do not call preloaded bursts
end-to-end latency. No software source/clock/driver/firmware substitution.

CPU implementations receive the same dense input bytes and emit identical two
counts: packed SIMD overlap, inverted connection index, and a fixed adaptive
selector (input popcount<=15 → zero result; <=64 → inverted; otherwise SIMD).
Selection uses observed input, never fixture density labels or expected outputs.
Weights/index setup is resident, excluded like native program setup; input packing,
selection and result calculation are included. Scalar portable fallback is labeled;
the Nano must use its compiled NEON path. Untimed numerical smoke then256 timing
repeats per density in three temporal batches. Keep all three algorithms and
fail on any output mismatch; no empirical choice of a new cutoff after results.
CPU uses one pinned core, same as host native actor. Capture host configuration.

## B. Quiet-host native learning

Use the exact previously qualified six-FNP, three-plastic-bank width512 recipe,
seed102101, full768-event repair stream. Copy/hash parent fixtures without
reopening its study. Serial teaching prefix512 events, recognition-only repair128,
then128 query/retention events. Parent initial/program/learning packets unchanged.

Repair chunks use the supported queue request chunk+1; preload all inputs/teachers
while the queue is held, then release. Mode remains recognition-only in each
chunk. Compare chunk8 polling versus chunk8 sleeping20ms. Only after both outputs
and full-state checks pass admit chunk32/sleep100ms, thenchunk128/sleep250ms.
Repeat chunk128 sleep twice if qualified. No alternate queue-flag override.
Failure to allocate/enqueue/complete at first wake stops that expansion and is
preserved. No hidden retry with a longer sleep or smaller chunk.

Record driver read/write counts and thread CPU/wall/context switches across the
released interval. Sleeping arms perform no driver accesses between release and
the first wake check; the completed descriptor ID must already equal the target
on first read. Polling arms may read completion. Verify every output and complete
bank RAM after each chunk with independent CPU reconstruction; query RAM stays
fixed. Save complete bank states after every serial event and chunk boundary.
Never run CPU neural updates before the corresponding native chunk completes.
Check all fixed memory before/after, and record whole-program cleanup.

This tests execution offload while one host thread blocks. It does not establish
all-host CPU sleep, interrupt-driven completion, live sensor autonomy, teacher
generation, physical simultaneous neurons, or energy savings. Fixed sleep gives
an upper bound on completion, not the actual hardware completion timestamp.

## Admission, stop and claims

Source/binary/fixture/protocol hashes frozen before hardware. Gates explicitly
admitted after independent raw-derived checks. Exclusive nonblocking owner lock;
device-free check, max40 sessions/20000 forwards/120s per owned actor and private
90-minute clock. Stop only owned jobs; preserve failures; no automatic seed,
implementation or task replacement. Post-close source hardening is labeled.

Publish native operations demonstrated, scaling, CPU comparison and gaps as
separate endpoints. A native operation can exist without an application win.
No CPU speed, power, all32-NPU learning or unique general intelligence claim from
mere device execution. A material advantage post requires its measured gate;
otherwise retain a technical evidence visual and mark that post gate unmet.
