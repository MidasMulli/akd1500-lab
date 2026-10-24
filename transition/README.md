# Learning changes a bounded native trajectory

External examples reassign two transitions in an engineered eight-state memory.
Native FNP3 connection changes then alter the sequence followed by output-to-input
feedback. The fixed second FNP3 decodes the predicted class into a state code;
the host does not select successive states or upload replacement weights.

Five initializations: **10/10 changed associations**, **30/30 unchanged associations**,
and **1,280/1,280 subsequent predicted states**. Frozen and unlabelled controls
each match **309/1,280** states of the new graphs and recall **0/10** changes.
All 26 diagnostic outputs and visible learning-SRAM states per successful session
match independent scalar and packed CPU implementations. The stronger eight-entry
CPU transition table also passes recall and rollout endpoints.

[Assessment](../studies/20261024-learned-transition/ASSESSMENT.md) ·
[Prospective primary protocol](../studies/20261024-learned-transition/PROTOCOL.md) ·
[Separate finite-replay protocol](../studies/20261024-learned-transition/REPLAY-PROTOCOL.md)

```sh
python -m transition.archive audit
python -m transition.audit_mutations
python -m transition.visual --out /absolute/output --video
python -m transition.video_qa /absolute/output
```

The portable audit needs NumPy; the renderer also needs Pillow and FFmpeg.
Hardware compilation/mapping requires the separately supplied Akida 2.19.3 SDK,
runtime and headers. No SDK binaries, headers or generated programs are distributed.
`prepare.py` never opens physical hardware. `build.sh` compiles the primary instrument;
`replay_build.sh` compiles the secondary instrument and cleanup tool. Environment:
`AKIDA_ENGINE_ROOT`, `AKIDA_RUNTIME_LIBRARY`, `FLATBUFFERS_INCLUDE`.

The saved study is closed. `run.py` retains source freezes, finite budgets, single-owner
locks, deadlines and stage gates. New physical reproduction needs a separate study
authority/state. Earlier failed CPU/control runs and both replay failures are archived.

## Boundaries

Eight binary records, eight teacher classes, one prototype per class; engineered
prior, not learned representation or discovery. Partial starting cues omit four
of 16 unique state features. Each seed cue is followed by 32 unlabelled, prequeued
steps in device system SRAM while the controlling worker sleeps for 5ms. The
queue ends. This is not indefinite self-scheduling, whole-host quietness, or a
CPU/energy advantage. Adaptive thresholds limit this to one changed-map episode;
arbitrary continued reversal remains unqualified.

The separate inference-only hardware-generated replay branch **fails before inference**
on its two-pass configuration. Its one-pass baseline passes. Duplicating unique
backing spans fixes an initial allocator error but does not fix the configuration
timeout. Four/eight-pass variants are never admitted. A compiler `loopitself` option
rejects this IP target; that rejection does not enumerate silicon capabilities.
Owned cleanup resets the failed installation and verifies the three DMA run bits
are clear. No inter-pass learning-state retention is claimed.

The film replays measured outputs and selected-connection memory. Class activity
is inferred through the one-to-one fixed decoder. Geometry, feature grouping,
camera motion, flow particles and playback timing are schematic, with no electrical
spike timings or physical neuron positions implied. The earlier high-address FNP2
learning issue remains open.
