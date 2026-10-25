# Prospective common-clock contention test

This amendment is frozen before its qualification and busy-device execution.
The original source/binary freeze remains unchanged. A separately compiled
`host_window` retains the exact arithmetic and learning rule, adds READY/start
and audit-release barriers, fixed-rate pacing and event timestamps. Its generated
source ancestry is in WINDOW_SOURCE.json. No deployed server/model is modified.
The original cutoff,80 physical sessions,8192 forwards and120-second child caps
still apply across both raw/ and window_raw/. No inference skipping.

## Preconditions

The full transport/completion screen passed exact outputs and captured complete
RAM boundaries. Across all three initialization seeds and rotated confirmation
repetitions1/2, baseline and selected100us-sleep, single-image, profile1 both pass
learning and frozen arms. Median-of-cell medians for learning:7.848ms/7.686ms
wall/caller CPU baseline,7.932ms/0.355ms candidate; fastest CPU OpenBLAS4
1.227ms/1.225ms. Selection and confirmation are distinct. The new claim being
tested is co-scheduling behavior, not prediction quality or electrical efficiency.

## New-source qualification, before contention

Freeze all new sources/binary, unchanged source freeze, fixtures/references and
external foreground sources/checkpoint. Run all five CPU implementations on the
new full=false path, learning and frozen, primary seed,64 events, no pacing.
Require every output and complete final learner RAM to match the frozen exact
trace. Then native8-query smoke, followed by64-event spin-learning,
sleep-learning and sleep-frozen. Verify initial/final fixed packets, no registered
host weights, exact output stream/final full RAM, device release. A mismatch stops
expansion; repair requires preserved source versions and a prospective amendment.

## Foreground work and measured window

Use the unchanged hash-checked Nano T202 checkpoint
3342f73cf939f689abfb1189faf3580719acbf05ca2d439001ee13887ea5bc65,
native CUDA BF16. Load a separate instance through streaming.nano.Nano, retaining
the deployed sources' checked() manifest checks. Fixed scene202610249100010,
paletteK25, twin0; observe once and freeze the starting belief. Each timed call
clones that belief, runs five learned steps for action[.15,.5] at times0..4,
decodes RGB and copies uint8 output to host. This is a repeated forecast workload,
not a planner, a natural stream, a new training result or an application-quality
test. Eight warmup forecasts must be bit-exact; all later output hashes must agree.
Model parameter digest must remain unchanged. Save observed/forecast images.

All arms use the same six-core Nano, CPUs2–5 for the foreground process.
Actor affinity:CPU2 for native and single-thread controls;CPUs2–5 for OpenBLAS4.
No clock, firmware, power, governor or deployed-service changes. Record thermal,
frequency, load and full /proc/stat bracketing snapshots; do not claim CPU
performance counters or exact DRAM traffic. Caller CPU and host busy ticks are
separate endpoints. Existing server remains running; activity is a limitation.

Start actor/model at a shared monotonic timestamp500ms after both READY messages.
64 actor events arrive at20Hz,50ms intervals, for a3.2s window. Do not drop or
skip events: late work remains recorded and misses its50ms completion deadline.
Eight prefix queries,32 teaching events,16 repeated queries,eight novel queries;
the actor sees the same synthetic RGB, labels and order as qualification.
Foreground runs continuously until the declared window end; report forecasts
completed inside it, median/p95 completion latency and process CPU. A call that
starts inside but finishes after the end is preserved and excluded from FPS.

No reference convolution, RAM snapshot, fixed-packet observer or file output may
run in the actor's operational loop. It records returned scores, event timestamps
and timings in memory. After WINDOW_DONE, parent waits for the foreground's final
call and weight check to finish, then releases AUDIT. Only then may actor read RAM,
reconstruct the exact CPU stream, compare outputs and serialize arrays. Save the
release timestamp and prove it follows the final foreground completion.

## Arms, order, replication and decision

Five arms:none(foreground only),cpu-blas4,cpu-blas1,native-spin,native-sleep100.
Exactly three repeated blocks, with actor seeds107501/107601/107701 respectively.
Block0 order:none,blas4,spin,blas1,sleep.
Block1 order:spin,blas1,sleep,none,blas4.
Block2 order:sleep,none,blas4,spin,blas1.
No concurrent compilation or other owned benchmark during these windows.
Each cell loads/warms a fresh foreground instance; all setup is excluded from
the common window. This is15 short windows,6 native sessions,384 native forwards.
The cost of reloading/programming is separately outside steady-state timings.

Report every arm/block. Success for a practical co-scheduling lead requires all
exact output/memory/weight gates, no actor deadline misses, and >=5% more foreground
completed forecasts for sleep100 than BOTH CPU paths in every repeated block,
without >5% worse foreground p95 than each CPU path. Failure to meet that gate is
reported as no demonstrated application-throughput advantage. Lower actor CPU
alone remains a qualified resource tradeoff. No energy claim is admitted without
the physical calibrated rig; no extrapolation to all-chip simultaneous learning.
