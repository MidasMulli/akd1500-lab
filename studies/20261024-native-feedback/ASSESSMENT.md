# Bounded native feedback qualifies; self-scheduling remains open

The AKD consumed its own preceding output as the next input over a pre-scheduled
32-step sequence while the Nano control worker slept. Native learning changed
local FNP3 connections during that sequence and improved later partial-cue
responses. This is a mechanism qualification, not an efficiency or useful-task
win. The study is closed: **22 physical sessions, 518 forwards, no owned jobs**.

## What actually executes

The Nano first configures a 128-input/128-record FNP3 learner and a second,
fixed FNP3 identity relay. Both binary input and output use 128-byte dense
payloads. The Nano uploads one seed, allocates separate device-SRAM output
buffers, and prepares a finite queue. Each descriptor after the first points
directly to the preceding output's payload, after its 32-byte output header.
Only that descriptor's input address is changed. The relay's entire fixed RAM
packet is independently read before and after; it remains unchanged.

After release the worker sleeps for 5 ms. Its driver records zero reads and
one release write, with no further writes during that quiet interval. At first
wake, all scheduled jobs have completed. Stored output payloads, every returned
binary response and the complete terminal learning SRAM agree with independent
Python and packed C++ references. Separate prefix runs at 1,2,3,4,5,8,16,32 steps
qualify connection trajectories; those are not simultaneous observations of
internal events during one quiet run.

This feedback travels through the on-device scratch SRAM and DMA job engine.
The learner's selected connections stay in its local FNP3 SRAM. It is not a
direct cyclic mesh route, an indefinitely repeating descriptor ring or a
neuron-generated scheduler. No host reads, transforms or copies intermediate
outputs during the released run. The Nano OS and interrupt handlers remain
active; only this control worker is quiet. Constant API class 0 and enabled
learning are fixed before the primary run.

## Learning result and controls

The transition ring is explicitly engineered: four groups of 32 active state
bits. Each learning record starts with 16 connections to its predecessor state
and 16 elsewhere. Output threshold 13 allows the ring to execute before any
learning. Learning threshold 15 admits a native update. Four winning records
replace 16 connections each, making 64 replacements per initialization. The
other 124 records do not change selected connections. All 128 records retain
exactly 32 connections.

Three frozen initializations produce the following 32-step outcomes:

| Arm | Sequence | Native connection replacements per seed | Later partial-cue responses across 3 seeds |
|---|---|---:|---:|
| Native learning + feedback | Cycles through all four states | 64 | 12/12 |
| Learning frozen + feedback | Same four-state cycle | 0 | 0/12 |
| Seed reused, no output feedback | Same output state repeated | 16 | 3/12 |
| Feedback severed after first step | One output, then all-zero outputs | 16 | 3/12 |

The later four queries contain half of a state cue, chosen before hardware
admission from features missing in the initial winning record. After native
learning, that record responds; the matched frozen model does not. There is no
host threshold adjustment or output correction. These query operations do not
change connection memory. They are new externally supplied queries after the
quiet loop, not further autonomous feedback steps.

This is seeded rehearsal and pattern completion. It does not learn the ring
topology, discover a new external rule, generate reliable outcome truth from
nothing, or demonstrate useful environmental prediction. The frozen arm proves
the programmed ring itself does not depend on learning. A CPU can implement the
same arithmetic and task; the exact packed CPU does so in approximately 1.9 µs
per step for the primary 32-step case, over 200 fresh repeats. The 5-ms native
quiet window is an observation bound and cannot be compared as device latency.
No speed, energy or CPU-resource advantage is established here.

## What stops when the schedule ends

After the queue drains, the worker sleeps again for 20 ms without driver I/O.
The completed-job register and complete learning SRAM remain unchanged at the
later observation. Raw output payloads were read after the first wake; they
were not reread after this second idle interval. We therefore claim stability
of completed jobs and learning SRAM, not a separately measured absence of all
electrical activity or all possible hidden state changes.

This demonstrates a bounded output-dependent loop while the host worker is
quiet. Extending the duration currently requires supplying more descriptors.
Indefinite self-scheduling, native delayed cyclic routing, external experience,
learned context formation and a useful efficiency advantage remain unresolved.

## Reproduction, attempts and visual

```sh
python -m recurrence.archive audit
python -m recurrence.visual --out /absolute/output --video
```

The portable archive contains the numeric fixtures, observed outputs, SRAM
readbacks, descriptors, quiet-interval counters, fixed-memory checks, source
hashes and CPU results. It does not redistribute vendor runtime, headers or
generated executable programs. Hardware code requires separately licensed
Akida 2.19.3 runtime and the user's device. The historical admission file is
closed; this report is not new execution authority.

No compiler or physical correctness failure occurred. The build warning from
the unchanged inherited benchmark entry point is preserved. Before admission,
an authoring review corrected the reference ring's array-roll direction and a
summary field that would have printed a Boolean instead of an interval. No
physical result was discarded or retried. The replay-controller alternative
was reviewed in runtime source; hardware admission used the simpler matched
dense SRAM payload path. No firmware, clocks or driver installation changed.

The 24-second 4K60/1080p60 film presents actual returned outputs and sampled
connection masks. Geometry, camera and playback timing are schematic. Connection
snapshots are explicitly labeled as independent prefix readbacks. The finite
queue exhaustion and subsequent host-supplied recall queries remain visible.
No electrical spike times or silicon die locations are inferred.
