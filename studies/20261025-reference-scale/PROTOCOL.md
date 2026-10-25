# Reference deployment and wider resident learning

Direct user authority: build the Nano reference deployment, then larger native
workloads against strong CPU controls; inspect each stage before expansion;
capture measured visuals and identify remaining power-rig work. Previous studies
stay closed. New two-hour ceiling, 48 physical sessions / 4096 forwards, at most
120 seconds per owned child. Shared lock and free-device check before opening.
Stop expansion on a failed numerical audit; preserve failures and prospectively
freeze any repair. Do not change clocks, voltage, firmware, or unrelated jobs.

## Reference first

Pin BrainChip DevHub e7e4a57de706bfabb417e38824bee186e875a6ae and verify the
Git-LFS SHA256 of its pretrained visual wake-word FBZ. Use the installed 2.19.3
SDK's normal Model.map/forward deployment, Minimal first, AllNps only after the
Minimal gate passes. Sixteen frozen synthetic RGB frames (zero, full, gradient,
seeded random) are a deterministic deployment check, not a dataset accuracy or
representative sparsity benchmark. Compare every returned output byte with the
SDK software backend. First 8 frames, then 64 repeated frames per mapping.
Record mapping, passes, model external-memory requirement, setup and per-frame
wall/process CPU times. The SDK software backend is a correctness comparator;
its timing is descriptive, not the strong CPU-performance control below.
No claimed pretrained accuracy, energy, application improvement, or learning.

## Larger native workload

Reuse the qualified resident_spatial C++ backend and resident_prepare.main:
32x32 RGB -> 32-channel HRC -> signed 4-bit spatial CNP filters -> signed 2-bit
projection -> 128-record FNP3 learner, 128 features / 32 connections per record.
Retain the exact 64-event schedule: 8 queries, 32 labelled teaching events,
16 repeated read-only queries, 8 novel read-only queries. Native learning and
matched trained-never/frozen initial twin. Labels and filters are synthetic.

Width 512 is the replication gate. Attempt 1024, then 2048, then 4096 filters,
seed107501. Compile/virtual exactness and one resident pass, local FNP3 RAM and
no host weight registration are required. Do not silently change projection,
precision, arithmetic or mapping policy to force a larger width to compile.
An unsupported width is a preserved boundary and stops width expansion.

All three CPU controls must first pass at each width: preallocated O3/native
NEON dense, NEON zero-skipping, and four-core OpenMP NEON zero-skipping. Pin
single-core/native caller to CPU2, parallel to CPU2-5. All must agree on every
feature, potential and full per-event learner memory. Then native learning and
frozen arms; verify every potential and all 1026 learner RAM words each event,
all fixed CNP configuration packets before/after. HRC physical state is outside
that configuration readback. Queries must not edit memory; learning must edit it.
8-event smoke before the first64-event native stream. Inspect both arms before
the next width. At the largest passing width, add seeds107601 and107701 only
after seed107501 passes. Three temporal repeats, with rotated mode/arm ordering,
are permitted for passing widths. Timings include input/queue/fetch and exclude
diagnostic readback/CPU verification; busy-poll caller CPU is included. Treat
initial eight events as warmup; report remaining56 events together and by phase.

Report actual mapped unique NPs, nominal dense MACs, actual configuration bytes,
connection swaps and exact CPU matches. Nominal MACs are workload arithmetic,
not counted hardware operations or proof of electrical neuron simultaneity.
Scaling/native capability and CPU advantage are separate outcomes. No power
or energy inference without the assembled, calibrated rig. Plot and animate
saved potentials/connection differences only; geometry/playback are schematic.
