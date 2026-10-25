# Reference deployment and larger native learning — October25,2026

**Reference deployment passes. Four times the image work passes with exact native
learning memory, but the strongest CPU control remains faster. Two compiler
boundaries stop further expansion of this architecture.** No energy result is
available; the physical power rig still needs parts and calibration.

## Deployment reference

On the Nano, installed Akida2.19.3 runs BrainChip's pretrained visual wake-word
FBZ from DevHub commit`e7e4a57de706bfabb417e38824bee186e875a6ae`. The downloaded
asset matches its Git-LFS SHA256
`90531d024b3359f836651748ff867aeaf0019aa0dd7067b9849fb5a3f5381b22`.
The reference uses normal SDK Model.map/forward, not our experimental MMIO actor.
Vendor weights/runtime are not redistributed here.

| Mapping | Neural positions, excluding HRC | Passes | Model external-memory requirement | Median frame latency |
|---|---:|---:|---:|---:|
|Minimal|16 + separate HRC|1|0bytes|7.923ms|
|AllNps|28 + separate HRC|1|0bytes|5.621ms|

Both mappings pass an8-frame gate and64-frame check,144physical forwards total.
Every returned score equals the SDK software backend. The bank has16frozen
synthetic frames, repeated for the64-frame test. These are **deployment exactness
checks**, not person-detection accuracy, representative natural-image sparsity,
vendor benchmark reproduction, or native learning. SDK software timing(~9.03ms)
is descriptive; it is not the optimized CPU control used below. Setup is separate
(about150ms Minimal,188ms AllNps). One pass and zero external model memory do not
establish that the entire host path is RAM-free or autonomous.

## Larger resident native graph

32×32 ->64×64 RGB quadruples nominal dense arithmetic,9.724M ->38.896M MACs per
frame. Both graphs use HRC ->two spatial CNP partitions ->projection CNP ->one
FNP3 learner in one resident pass. The128-record learner grows from128 to512
candidate feature bits; each record still selects32connections. The full visible
learner packet increases from4,104 to10,248bytes, read back after every event.
No host weight memory is registered by the native actor. All fixed CNP packets
match before/after; HRC configuration is outside that readback routine.

At64×64, three fresh initializations and both learning/frozen arms pass, including
two additional isolated timing repetitions. Native output potentials and every
learner-memory word match all five independent CPU execution paths. CPU feature
bits match independent integer convolution. Intermediate native feature bits are
**derived/checked end-to-end, not directly probed electrically**. Each stream has
8initial queries,32labelled teaching events,16repeated read-only queries and8new
read-only queries. There are8connection-edit events per learning stream; repeated
presentations can leave memory unchanged. Frozen/query states never edit RAM.

Labels and filters are synthetic. Shapes have separately generated seeded banks;
they are not two resolutions of a shared natural scene. This establishes native
execution, residency and scaling, not useful vision accuracy or a new memory policy.
The28-NPU pretrained reference and four-NPU custom learning graph are separate.

## Strong CPU controls and operational timing

All CPU paths include image processing, feature packing and the exact native
binary learning rule. Fixed-weight preprocessing is setup. Controls are C++ O3,
NEON dense, NEON zero-skipping, four-thread OpenMP/NEON, and one/four-thread
OpenBLAS0.3.27 SGEMM. BLAS includes per-frame im2col packing. Its float32 integer
sums remain exact at the bounded magnitudes, with every result/state checked.
Single-thread/native caller pinned toCPU2; four-thread paths toCPU2–5.

| Learning arm, median ms |32×32|64×64|
|---|---:|---:|
|AKD input/queue/completion/fetch|2.265|7.857|
|CPU NEON dense|2.740|11.423|
|CPU NEON zero-skipping|2.257|9.525|
|CPU OpenMP/NEON,4threads|0.646|2.635|
|CPU OpenBLAS,1thread|0.937|3.288|
|**CPU OpenBLAS,4threads**|**0.324**|**1.198**|

The strongest CPU path is about7.00× faster at32×32 and6.56× at64×64. These are
observed full-path comparisons, not a proof of the fastest conceivable CPU or
AKD implementation. Native completion busy-polls; it does not free/sleep the
whole host. Full RAM audits and CPU diagnostics are outside operational timing,
but included in saved observed runs. No pure-silicon or energy ratio is inferred.

Only isolated, rotated process repetitions1/2 enter this table and the figure.
Initial qualificationr0 remains separate. One initial seed107601 CPU qualification
overlapped virtual construction of seed107701; those timings are explicitly
excluded. Subsequent repetitions ran serially with no owned compiler alongside.
Three CPU streams per process and one native stream per process; after8warmup
events there are112native timed events at32×32 and336 at64×64. The respective
CPU counts are336/1008. Events within one seeded stream are correlated; these
counts are not independent generalization samples. No clock/power settings changed.

## Preserved boundaries and corrections

* 1,024filters: the signed2-bit projection cannot map its input filter into a
  CNP; no physical run. The2,048/4,096filter expansion was not attempted.
* 96×96: compiler rejects stride>1 with max pooling when spatial partitioning
  requires multiple NPs; no physical run.128×128 was not attempted. These are
  **compiler/topology boundaries, not silicon maxima**.
* Optional sysfs permission/no-data errors and a JSON tuple/list mapping-comparison
  defect stopped early admissions. Mapping comparison was repaired and rerun;
  the failed physical attempt sent no inference frames and released the device.
* One CPU stream initially lacked its remote auditor dependency. Its successful
  saved arrays were re-audited after copying the existing helper, without rerunning
  the computation. An archive attempt begun before rsync completed also failed;
  its logs remain, followed by a successful complete reconstruction.
* Initial BLAS4 stdout incorrectly labelled the thread count as1; its own BLAS
  configuration log and affinity recorded4. Only that stdout expression changed
  before isolated repetitions; arithmetic was unchanged. Old receipts are retained.

All32successful physical sessions/1,824forwards are in the portable numeric archive:
28custom-graph sessions/1,680forwards plus4reference sessions/144forwards.
The pre-inference failed admission is additional and preserved.154matched custom
CPU/native streams are reconstructed, including preliminary512-filter replication.
Across native custom streams,1,077connections were added and1,077removed; this
includes repeated identical initializations, not1,077unique discoveries.

## Visuals and next decision

[4K results](RESULTS.png) and the recorded larger-stream replay show actual
returned potentials, actual visible connection changes and a matched frozen
chip.512feature inputs are derived from independently verified integer math;
camera, cables and coordinates are schematic. They are not electrical spike-time
measurements. The video shows eight sampled records from128, not all chip neurons.

The next discriminator is calibrated card/whole-system energy under an admitted,
useful sparse pretrained workload, followed by controlled host-contention tests if
correctness passes. Rewriting an architecture to cross the compiler boundaries
is a separate prospective study (for example, separating pooling from strided
convolution), not a continuation silently relabelled as this test.

The user confirms only a spare connected ESP32-S3. Remaining hardware and checkout:
[power-rig finish list](../../power/FINISH.md). Software power tests pass, but there
is no assembled/current-calibrated rig. This is a research baseline and honest
capacity update, not the promised practical-advantage X milestone.

Recheck with`python -m coprocessor.extent_audit audit`. Numeric archive SHA256:
`62f3cdef2e76d301e3365c13e3f080a9db7ffba79feda5373400d483a851a716`.
See[prospective protocol](PROTOCOL.md),[extent amendment](EXTENT_AMENDMENT.md),
[per-process timing CSV](TIMINGS.csv) and[computed summary](SUMMARY.json).
