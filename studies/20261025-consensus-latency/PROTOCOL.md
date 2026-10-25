# Prospective agreed-update latency benchmark

Direct user authority: build and run on Nano; compare native AKD against both CPU
controls, separate transfer/verification overhead, and time reliable updated recall.
Parent studies stay closed. The inherited seven-FNP3 program and all three seeded
initializations are byte-identical to the qualified consensus-memory study.

## Frozen workload and endpoints

Seeds 124101, 124201, 124301. Four cues, two revised per cycle, two unchanged.
The original corruption / protection / delayed corroboration / accepted revision
schedule is retained. After each accepted teaching transaction add 16 read-only,
source-answer-masked queries of that cue. No new teaching information is supplied
to these queries. The original shuffled final exams remain. Gate: one cycle; repeat:
six reversals. Every run starts with a freshly initialized program, never restored
partway through learning. No sparse-execution change is made to the chip program.

Reliability is operationally 16/16 consecutive teacher-silent recalls of the changed
cue and no subsequent error on that cue within the same cycle. This is a bounded
repeatability criterion, not a statistical guarantee under arbitrary future data.
Report teaching attempts, learning-path active time, confirmation-query time,
active time until confirmation, and observed elapsed time including diagnostics.
Learning-path active time excludes inserted diagnostic queries and full-RAM audits;
it includes necessary source queries and other cue transactions in sealed order.
Readiness starts immediately before the first accepted-transfer source query, after
the pre-existing agreement probes have completed. Source-grounding/acquisition time
is outside this endpoint and is reported separately by event phase if needed.

## Implementations

1. AKD: unchanged resident native network, card-SRAM carry, constant transfer class.
2. CPU replica: existing packed binary learning algorithm, with fixed routing
   evaluated through precomputed sparse coefficient lists. All output bytes and
   complete learning RAM must match the independent Python replay.
3. CPU agreement: optimized fixed-size cue/source/student arrays implementing the
   same functional policy, without a neural simulation. It receives source evidence,
   cues and phases, never evaluation truth. Also report a batched, no-per-event-timer
   estimate because clock-call overhead can dominate this tiny CPU operation.

All paths run on the Nano, pinned to CPU2, O3/native compilation. Setup, program
loading, reference generation, allocation and disk writes stay outside operational
timings. CPU fixed-route preprocessing is setup, not charged per transaction.
No device CPU clock/power configuration or unrelated jobs are changed.

Native components: mode configuration; raw cue/phase input writes; descriptor
submission; queue release write; release-completed to observed status-completion;
output fetch. Completion observation includes PCIe polling and is an upper bound
on resident execution, not a pure silicon timer. Input, descriptor, release,
completion polling and fetch cannot be mathematically stripped into pure PCIe vs
compute; their observable boundaries and byte counts are reported directly.
No host writes occur while the queued job executes. Full learning-RAM readback
and comparison are timed separately. No file I/O occurs within the event loop.
CPU event clocks are reported raw along with clock-call calibration; do not
subtract unstable timer estimates. Batched CPU agreement timing is separate.

## Stages, admission and stop rules

Software reconstruction and meaningful audit/mutation tests precede hardware.
First run seed124101 with per-event full-bank snapshots for CPU replica and AKD.
Exact outputs, every saved state and protected/updated recall must pass. Then run
the same gate with only final full-state readback and require exact outputs/final
states. Finally run three seeds x three timing repetitions x six cycles. Rotate
the serial order of CPU agreement, CPU replica and native arms by repetition.
Every fresh process replays the same initial learning prefix before timed revisions.
Any failed native correctness gate stops expansion pending inspection; preserve
every build/run/admission failure. No automatic retry. Maximum 16 physical sessions,
80,000 physical forwards, two-hour scientific bound; only owned jobs are stopped.

Shared device lock plus free-device check. Source/binary/fixture hashes frozen before
each process; UTC times, CPU affinity/frequency/governor/temperature and exit state
saved. Report median/p95 and sample counts, without pretending correlated repeated
queries are independent experiments. Do not claim energy advantage, maximum chip
throughput, independently clocked learning groups or CPU impossibility.
