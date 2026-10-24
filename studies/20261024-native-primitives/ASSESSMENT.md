# Native execution paths and blocked-host learning

Closed October 24, 2026 Central. **30 experimental device sessions plus one
zero-forward allocation/cleanup session**, 7,984 verified native forwards and
7,404 complete learning-bank RAM captures. The failed 128-event learning stage
is preserved, including its 512-forward verified prefix and cleanup crash.
All owned jobs are stopped/completed and the device handle is released.

## Same binary computation on two native paths and three CPU implementations

Every graph uses the same 512 binary input bytes, 32 positive connections per
output, threshold 15 and two group-count results. FNP and CNP weight matrices are
byte-identical at matched widths. Ordinary compiler output maps in one resident
pass; no FNP2, custom learning adapter or source-policy substitution is used.

Three temporal native sessions per cell reproduce the half-active counter
medians. These are one frozen weight/input bank on one chip, not independent
worlds, seeds or a manufacturing sample. The table uses 256 active input bits:

|Detector outputs|FNP service cycles|CNP service cycles|CPU packed NEON µs/input|Native CNP serial host µs/input|
|---:|---:|---:|---:|---:|
|128|8,919.00|16,202.50|0.70|69.92|
|256|11,487.00|16,223.00|1.26|69.92|
|512|16,623.00|20,827.50|2.40|83.11|
|1,024|—|20,880.50|4.67|83.14|
|2,048|—|22,504.00|9.20|86.96|

The 128→2,048 CNP population grows 16-fold while monitored service grows only
1.39-fold, with the largest graph using 11 compute CNPs plus a fixed FNP readout.
This supports useful aggregate processing capacity across mapped resources.
It is not a per-neuron same-clock trace or arithmetic-only timing. Cross-FNP/CNP
readout placement differs; matched mathematics does not isolate one transistor
or route-level cause. Separate causal interventions qualify all 11 compute
parameter regions in `20261024-cnp-contribution`.

CNP service is activity-sensitive: at 128 outputs it changes from 4,037 cycles
with zero active bits to 29,082 with 512 active bits. The FNP path is much flatter.
All host input transport here remains dense. Native event-dependent work does
not establish sparse host ingress or a CPU/energy win.

The CPU controls are compiled packed NEON overlap, an inverted connection index,
and a prospectively fixed adaptive selector. All reproduce every result. Dense
input packing and count calculation are timed; resident weight/index setup is
excluded like native program loading. Three temporal batches give 1,843,200 timed
CPU calls. CPU is faster on this workload through the measured native serial
host path. This does not establish the fastest possible native implementation,
chip-only latency, all-chip peak, or a universal CPU advantage on other formats.

## Native learning continues while one host worker blocks

The already qualified six-FNP graph has three 512-record plastic populations
and two fixed encoders plus readout. Fresh full prefixes restore hidden learning
history; host labels/modes remain prescribed. Recognition-only repair 128 events
is queued in chunks, followed by 64 changed-signal and 64 old-signal queries.

|Repair treatment|Complete output/RAM check|Released-interval caller CPU across 128 teaching events|
|---|---|---:|
|8-event chunks, polling|Pass|21.831 ms|
|8-event chunks, sleep 20 ms each|Pass|0.126 ms|
|32-event chunks, sleep 100 ms each|Pass|0.035 ms|
|128 events, requires 129 queue slots|Failed before repair|Not an admitted result|

Sleeping intervals have zero driver reads, one queue-release write, positive
voluntary context switches, and a completed target descriptor on the first wake
check. All outputs and captured full RAM match polling/reference. Both downstream
memories stay unchanged during recognition repair; old and changed-signal queries
score 64/64 each. This is one native initialization and one run per qualified arm.

The matched 8-event comparison reduces released-interval caller CPU by 99.42%.
That excludes input preload, queue preparation, output drain and diagnostics.
The fixed sleeps delay observation; they are upper bounds on completion, not
actual hardware completion timestamps or latency/energy improvements. This is
one blocked host thread, not all-host sleep, live sensor autonomy, new teacher
selection or a closed self-teaching system.

## Preserved failure, layer diagnosis and safe source publication

The 128-event stage requests 129 slots to retain one spare in the held producer
ring. Its capacity guard fails after the 512-event prefix and before any repair
event. The process subsequently exits with SIGSEGV during exception cleanup.
The recorded prefix independently matches every output and 1,536 full bank
captures. No successful 128-event learning result exists.

The installed runtime's source selects a 128-slot single-pass ceiling and clamps
larger requests. A separate zero-forward diagnostic physically measures
129→128, checks unchanged parameter memory, then cleanly unprograms/closes the
owned device. That is a runtime limit, not a measured silicon maximum. No
127-event fallback, full-ring 128-event retry or alternate flag was admitted.

Source review also finds a caller-buffer lifetime hazard in the executed actor:
its program vector is destroyed before catch-path cleanup, while the runtime
retains the caller's address. The exact SIGSEGV cause is not uniquely isolated.
`src/quiet_relay.cpp` is hardened after closeout to retain storage through cleanup
and reject the unqualified 128-event case before device access. It compiles and
its unsupported-case refusal is checked without hardware. The exact executed
v1 remains in `capability/executed/quiet_relay_v1.cpp`, bound by its original SHA.
Do not describe the hardened source as a successful physical rerun.

A nested-entrypoint source-build failure and premature pre-admission request
were stopped before hardware access and preserved separately.

## Verification and presentation

```bash
python -m capability.archive audit-public
python -m capability.contribution_archive audit-public
python -m unittest discover -s tests -p 'test_native_primitives.py' -v
```

Complete captured RAM is published as lossless changed-word deltas with the SHA
of every original complete snapshot. No per-event RAM observation is implied
inside a queued chunk; only its actual captured boundaries are measured. Numeric
archives, source versions and plots are public; vendor runtime/programs remain
external. The 24-second 4K60 replay separates the fixed and learning experiments.
Resource positions/connection angles are logical geometry, not die photography
or electrical spike traces. The CPU/practical-advantage post gate remains unmet.
