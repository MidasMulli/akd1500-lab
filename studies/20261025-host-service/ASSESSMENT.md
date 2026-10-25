# Native learning with less host CPU

**The host-service repair works; a forecast-throughput advantage does not pass.**
Sleeping 100 µs between nonblocking completion checks preserves every returned
value and native learning state while reducing caller CPU by 95.38% against
our spin baseline. Answer latency increases 1.07%. The strongest CPU comparator
still answers about 6.5× sooner. Native learning consumes less host CPU, but
this particular co-scheduled Nano forecast workload does not turn that resource
tradeoff into a qualified throughput win.

## Confirmed resource tradeoff

64×64 synthetic RGB → HRC32 → CNP512 → projection 8 → 128-record FNP3 learner;
32 selected connections per record, 512 feature inputs. Fixed processing and
learning memory reside on the card; no host weight memory is registered.
This is a custom characterization graph, separate from the pretrained visual
wake-word reference. It has not been trained or scored for a useful vision task.

Median of six learning-cell medians: three initialization seeds × two rotated
confirmation repetitions 1/2. Repetition 0 was the selection screen and is excluded.
Timing includes input processing/copy, enqueue, completion and consumer output.
Setup, RAM observers, reference reconstruction and file output are excluded.

| Path | Answer latency, ms/image | Process CPU during call, ms/image |
|---|---:|---:|
| AKD, spin completion |7.848|7.686|
| AKD, sleep 100 µs |7.932|0.355|
| CPU, OpenBLAS4 |1.227|1.225|
| CPU, OpenBLAS1 |3.525|3.516|
| CPU, OpenMP4 |2.626|10.300|
| CPU, zero-skipping NEON |9.557|9.547|
| CPU, dense NEON |11.426|11.413|

The sleeping path uses 70.99% less per-call process CPU than OpenBLAS4 here.
Process CPU includes that process's threads; it is neither whole-host CPU nor
joules. CPU worker spin between calls is outside this metric. That distinction
matters: OpenBLAS4 shows substantial between-arrival thread activity in the later
scheduled workload. Those counters are separately preserved.

Binary outputs preserve their declared thresholded contract but barely change
baseline latency. DMA writes are slower in this screen. Eight-image queues can
reduce amortized cost slightly, but hold a group for roughly 61 ms; that is not an
individual-image readiness measurement. Sleep 500 µs reduces caller CPU further
with a greater latency cost. The responsive sleep 100 µs/single-image setting was
selected prospectively in SELECTION.json before confirmation.

## Busy Nano: useful-resource lead, throughput gate failed

A separate hash-checked T202 instance repeatedly executes five learned steps,
decodes RGB and copies the forecast to host. It starts from a fixed observation/
belief; every forecast is byte-identical and model parameters stay unchanged.
This is a repeated inference workload, not natural streaming accuracy, planning,
learned integration of the two models or a new world-model result.

Each actor receives the same 64 learning/query events at 20 Hz, in a common 3.2 s
window. No events are skipped; every actor meets every 50 ms deadline. Reference
work/readbacks wait behind an explicit release barrier until the foreground's
last call finishes. Returned values and complete final native RAM match exactly.

| Foreground forecasts completed inside 3.2 s | Block 0 | Block 1 | Block 2 |
|---|---:|---:|---:|
| Model alone |130|130|133|
| + CPU OpenBLAS4 |127|127|126|
| + CPU OpenBLAS1 |132|129|129|
| + AKD spin |130|133|133|
| + AKD sleep 100 µs |132|129|130|

The prespecified gate required at least 5% more foreground completions than
**both** CPU paths in **every** block, with no more than 5% worse p95 latency.
It fails. Relative to OpenBLAS1, the sleeping path ties twice and improves 0.78%
once; relative to OpenBLAS4,it improves 1.57–3.94%. Do not call this a qualified
application-throughput advantage or a claim that CPUs cannot learn.

These measurements do not isolate DRAM contention, GPU bandwidth or individual
kernel scheduling as the cause. No memory-bandwidth counter or electrical power
channel was measured. The existing workload and available CPU cores may simply
leave too little useful contention for this saving to raise throughput.

Median summed operational actor CPU over 64 events is 24.32 ms for sleep 100 µs,
492.41 ms for native spin, 216.75 ms for OpenBLAS1,and 792.36 ms for OpenBLAS4.
Whole-process /proc ticks and whole-host busy ticks are also saved. Their actual
brackets last about 4.63 s and include the foreground's final weight check; they
must not be relabeled as exact 3.2 s counters. USER_HZ was directly checked at 100.
The existing live server stayed running; no clock/governor/runtime was changed.

## Evidence and limitations

80 physical sessions, 4,672 physical forwards; 208 primary/confirmation cells and
51 passing window cells. Full-audit single-image cells capture complete learner
RAM after every event; batch 8 captures it after each batch. All returned values
are checked. Deferred-audit windows capture initial/final state only and do not
claim per-event native RAM observations. Fixed CNP packets remain unchanged.
Three seeds are different initializations on one card, not independent hardware.

The portable archive reconstructs all spatial features with integer NumPy
convolution, reproduces canonical learning with an independent Python selector,
checks every saved output/full captured RAM boundary and source-freeze binding,
and verifies unchanged fixed packets and foreground-frame hashes. The nine new
semantic tests reject altered scores, memory, missing boundaries, overlapping
timestamps and an early observer release. The full repository suite passes 227
tests. Vendor programs/runtime/checkpoints are not redistributed.

One foreground-only setup timed out before READY; no AKD was opened and no
scientific window started. It is retained. The 90 s setup allowance remains under
a 120 s total owner cap. Pre-device source/name repairs and the preliminary offline
auditor's binary-header mismatch are preserved in failed_attempts/ inside the
numeric archive. They are not numerical hardware failures. The installed PyTorch
2.14.1+cu130 emits an Orin-support warning; actual GPU warmups, forecast outputs
and weights pass this bounded workload, not broad runtime certification.

Scientific execution closed 2026-10-25 16:54:14 UTC at the 80-session ceiling,
well before the 18:04:30 UTC cutoff. Device free; no owned actors remain.

## Next decision

Keep this low-polling transport as the qualified deployment baseline. Hold a
major superiority post: we have a measured resource tradeoff, not the practical
win sought. Finish the calibrated physical power rig before energy claims;
software CPU clocks cannot substitute for card/host energy measurements. A useful
learning task still needs a separate accuracy/generalization gate. Earlier
streaming quality failures and inference-skipping restrictions remain unchanged.

[Measured figure](RESULTS.png) ·[confirmation cells](CONFIRMATION.csv) ·
[all busy windows](BUSY_WINDOWS.csv) ·[protocol](PROTOCOL.md) ·
[common-clock amendment](WINDOW_AMENDMENT.md) ·
[portable audit](../../results/20261025-host-service.audit.json) ·
[reproduction](../../coprocessor/HOST_SERVICE.md)
