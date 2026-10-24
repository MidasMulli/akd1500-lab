# A native representation rewrite has a measured crossover

One physical AKD1500, 16 independently programmed sessions, 2,624 forwards and
80 before/after parameter-region readbacks. Exact outputs, declared fixed RAM,
idle/config counter controls and resident suffix outputs passed all audits.
The four smoke sessions preceded screen admission; the screen preceded two
temporal repeats with alternating path order. The original source, fixtures and
instrument binary are preserved by the 55-file execution freeze and a private
verified archive. Public analysis was added after closeout.

## What changed

The direct graph uses 512-input detector records, with 32 selected connections
in the first 128 positions. Its input domain contains four identical 128-bit
copies. The rewrite adds a fixed 512-to-128 four-copy consensus NPU ahead of
128-input detector records. Latent detector weights and threshold15 are the
same. Both graphs return the same two group counts. The host sends the same 512
dense uint8 bytes in both arms, without host compaction or intermediate reads.
The equivalence claim is confined to this synthetic four-copy domain.

Detector placement FNP3(2,1,0) and readout FNP3(3,3,0) are held fixed; the rewrite
adds FNP3(1,3,0). This is supported native graph composition with an extra NPU,
not arbitrary code on the NPU and not a new learning engine. Learning is off.

## Repeated native service result

The three full-session medians below are identical for each cell. Each density
contains 48 inputs per session. Percent reduction means less monitored service;
negative means the rewrite is slower.

| Detector records | Latent active bits /128 | Direct cycles | Compacted cycles | Reduction |
|---|---:|---:|---:|---:|
|256|0|11,498|13,596|-18.25%|
|256|64|11,490|13,578|-18.17%|
|256|128|12,009|15,037|-25.21%|
|512|0|16,634|15,660|5.86%|
|512|64|16,626|15,642|5.92%|
|512|128|17,145|18,122|-5.70%|

At width 512, half-active host write/enqueue/fetch medians are about 71.78–71.79µs
direct versus 69.89–69.92µs compacted. At width 256 they are 58.53–58.56µs direct
versus 64.21–64.23µs compacted. Host elapsed and DMA service are separate endpoints.

The paired eight-job suffix has **zero active input bits**. Its queued monitor
cost is 83,917 direct versus 82,291 compacted at width 256 (1.94% reduction), and
125,005 versus 98,691 at width 512 (21.05%). Three paired suffix repetitions in
each of three full sessions reproduce those counts. Corresponding width 512
release-to-observed-completion time is 319.18–320.01µs versus 252.97–253.64µs.
This excludes input preload, descriptor construction and output drainage. It is
not nonzero-activity throughput or complete application latency. The serial
suffix includes per-job fetch; its timing boundary differs from queued release.

The frozen >=5% half-active serial and paired burst service gate passes at 512
records in all three full sessions and fails at 256. That is a protocol outcome,
not proof that the rewrite helps arbitrary data or a useful application.

## Memory and limitations

Captured parameter payload, excluding two synthetic read-request header words
per region, is 20,576→18,528 bytes at width 256 and 41,120→26,784 at width 512.
The latter is 34.86% less, including encoder parameters. These are captured
parameter regions, not total chip SRAM occupancy. Packet headers identify the
requested target; they do not discover physical SRAM coordinates.

DMA monitor scope is buffer service rather than isolated arithmetic cycles.
No physical spike/current trace, CPU advantage, calibrated power measurement,
arbitrary instruction loader or improved internal learning algorithm is shown.
The extra NPU, redundant synthetic domain, fixed weight seed, single physical
chip and fully-active slowdown materially limit the claim.

Next: test a useful resident representation with matched CPU task alternatives,
and include nonzero queued traffic before generalizing the burst result. Keep
the failed learning-policy studies separate and preserved.
