# Resident recognition and routing with selective native repair

Two learning populations have different jobs and different teachers. Recognition
learns which noisy beacon is present. Routing learns which destination that
identity means. Both weight stores remain resident in the same single-pass
AKD1500 graph. The host supplies raw binary sensor events, labels and learning-
mode changes; it never supplies the intermediate identity message.

The graph occupies **five mapped FNP3 NPUs**, with **128 configured learning
neurons** and three fixed stages:

1. Recognition: NPU `(1,3,0)`, 64 learning records, four identity groups.
2. Fixed pairwise support comparison: NPU `(2,1,0)`.
3. Fixed categorical encoder: NPU `(2,2,0)`, one 128-bit feature block per identity.
4. Routing: NPU `(2,3,0)`, 64 learning records, four destination groups.
5. Fixed four-count destination readout: NPU `(3,1,0)`.

The categorical message is supplied network design. It is not learned search,
arbitrary recurrent wiring or electrical spike telemetry. The two learning
banks use the same native algorithm and different phase-gated teacher labels.
This does not establish simultaneous independently arriving input streams.
The native comparator chooses the lowest identity index when support counts
tie, including when no recognition neuron fires. That default is shown in the
replay and is part of the supplied decoder, not a learned confidence judgment.

## Measured result

After shared identity and destination teaching, every arm routes 64/64 held-out
old signatures correctly. A new signature set reduces this to 16/64. The same
patterns have new noisy presentations; the intended destinations stay fixed.

| Treatment after signature change | New signatures | Old signatures retained |
|---|---:|---:|
| Teach recognition only | **64/64** | **64/64** |
| Freeze both memories | 16/64 | 64/64 |
| Teach routing only | 16/64 | 48/64 |

This repeats for seeds **97101, 97102 and 97103** on one physical chip. They are
three initializations of one synthetic sensor world, not independent physical
devices or three separately sampled applications. The two noisy query banks
are distinct from the teaching presentations. Old signatures get no teaching
after the change. Routing's entire captured learning RAM stays byte-identical
through recognition-only repair. Every control replays the same native prefix,
including selector history, before treatment changes. No visible-RAM restore
substitutes for complete prefix replay.

The ten sessions contain **9,280 native forwards**, **37,120 measured output
counts** and **18,560 complete per-event learning-bank snapshots**. A separate
Python reconstruction checks every output and state against the C++ inline
reference and the native readbacks. Fixed message/readout RAM is checked before
and after each run. All task gates pass; admissions are closed and the device
and owner lock were released.

The exact native-rule CPU model also matches. A separate post hoc CPU comparator
using two binary exemplars per identity and a learned destination table likewise
gets 64/64 new and old probes after repair. This is a **native modular-learning
mechanism and synthetic functional result**, not a CPU speed or energy win.

## Public checks and visual

```bash
python -m pip install numpy pillow
python -m relay.audit_public
python -m unittest tests.test_signal_relay -v
python tools/visualize_signal_relay.py --video
```

The video is **3840×2160, 60 fps, 24 seconds**. It uses actual recorded input
bits, output counts and RAM edits from seed 97101. Dot responses are independently
reconstructed, not electrically measured. Flow motion and NPU/neuron geometry
are schematic and time is expanded. The right-hand memory stays fixed during
recognition repair. The 32-address grid shows a measured changed or maximally
responding recognition neuron; gold addresses are actual inserted connections.
The native outputs of matched controls are compared at the same original event
indices. Input colors do not encode identity.

Numerical traces are in
[`2026-10-23-signal-relay.npz`](../results/2026-10-23-signal-relay.npz),
with the protocol, placements, gates and receipt hashes in its JSON companion.
`archive.py` exports owned numerical evidence and never redistributes the
separately licensed compiler, runtime or generated executable programs.

## Native reproduction boundary

The completed dated producers intentionally refuse further native admission.
For a new study, prospectively version the protocol and bounded execution
envelope, compiler outputs, source freeze and explicit stage admissions.
`prepare.py` uses Akida 2.19.3 and the previously checked intermediate-learning
adapter to retain compiler routing. `relay_bench.cpp` checks every bank after
each event. `run.py` verifies sources, caps and ownership before opening the
device. `audit.py` independently reconstructs the raw records before expansion.

The initial compiler input-dtype error, missing dependency in the initially
selected environment, and incorrect C++ include path are preserved. Repairs did
not change the task, learning parameters, initialization or acceptance gates.
No installed driver, firmware, device clocks or unrelated process was changed.

Next questions are a controlled communication interruption, useful branching
and richer messages, resident bursts, model-size/activity scaling, and calibrated
whole-system energy against optimized CPU execution. None follows automatically
from this small successful task.
