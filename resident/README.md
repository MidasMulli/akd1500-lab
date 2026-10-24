# Three connected native learning memories

**Repair recognition while preserving two downstream memories.** The latest
FNP-only graph uses six resident NPUs: three plastic populations, two fixed
class-presence encoders and one fixed count readout. Each population allocates
512 neuron records. Across three initializations, recognition-only repair reaches
**64/64 changed-signal decisions and 64/64 old-signal retention**; frozen controls
remain at **16/64** on changed signals.

Complete RAM shows four additional existing recognition records changing during
repair, while every word of the original four learned records stays unchanged.
The host supplies inputs, teachers and mode changes; the native rule chooses
winning records/connections. Intermediate processing stays on the chip.
Connections actually changed in **8 / 4 / 4 records**, not all 1,536 allocated
records. This is one synthetic four-class world, not general autonomous learning.

![Three native memories, selective recognition repair](../visual/fnp-or-relay-4k.png)

[4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/fnp-or-relay-4k60.mp4) ·
[assessment](../studies/20261023-fnp-or-relay/ASSESSMENT.md) ·
[frozen protocol](../studies/20261023-fnp-or-relay/PROTOCOL.md) ·
[review-only X draft](../visual/fnp-or-relay-X-DRAFT.txt)

All **6,672 native forwards and 20,016 complete learning-RAM snapshots** pass
independent reconstruction. Complete snapshots are published as lossless
changed-word deltas with each original full snapshot's SHA256. The film is a
recorded, time-expanded replay. Intermediate responses are reconstructed from
measured state; geometry and flow are schematic. No electrical spike, CPU-speed
or calibrated energy advantage is claimed. The wrong-bank arm receives no
eligible repair updates, so is not a matched-plastic-work comparison.

Hardware-free verification (NumPy required):

```bash
python -m resident.or_archive audit-public
python -m resident.or_recruitment
python -m unittest discover -s tests -p 'test_or_relay.py' -v
```

`or_design.py` defines the task and reference; `or_prepare.py` is the qualified
producer for this frozen architecture; `or_run.py` and `src/or_bench.cpp` are the
as-executed physical actors. Use **or_archive.py** for the public delta archive.
`or_recruitment.py` independently checks record preservation; `or_visual.py`
builds the evidence replay with Pillow and FFmpeg. Physical reruns need a new
bounded authority, source freeze, stage admission and exclusive lock. Saved
campaigns are closed. Vendor SDK/runtime/programs are external dependencies.

## Preserved mixed-handoff failures

The successful FNP-only architecture is a separate recipe. It does not fix the
mixed FNP-to-CNP handoff. Thirteen fixed-inference sessions / 824 forwards show
that source, all-FNP and direct-CNP controls preserve vectors, while the tested
mixed identity graph collapses every nonempty vector to channel 63. The measured
mixed-comparison outputs follow that collapse model exactly. A two-bit source
candidate fails its 0/8 development smoke; no sealed native qualification runs.

[Contract assessment](../studies/20261023-handoff-contract/ASSESSMENT.md) ·
[single-layer compiler probe](../studies/20261023-distributed-plastic/ASSESSMENT.md)

```bash
python -m resident.contract_archive audit-public
```

The seven-cell single-layer compiler probe made no hardware forwards. Larger
cells request FNP2 under the tested Minimal/FNP3-only policy and are rejected;
that is not a silicon limit. `contract_design.py`, `contract_prepare.py`,
`contract_candidate.py`, `contract.py` and `src/contract_probe.cpp` preserve the
ordinary-compiler endpoint investigation and failed candidate.

## Earlier scale-up boundary


The new mixed FNP3/CNP route **did not qualify**. It compiler-maps six learning
banks (384 records) on 17 NPUs, but two-bank learning-state checks stop expansion.
Three one-bank mechanics runs pass; three mixed-bank adapter attempts fail.
The earlier verified two-bank FNP-only study remains valid and separate.

A separate inference-only follow-up isolates the handoff: the all-FNP and
host-fed CNP controls each pass128/128 patterns, but the unmodified compiler
FNP→CNP graph fails its first pattern (expected class0, measured class3).
That diagnostic adds5 sessions/321 forwards and leaves parameter RAM unchanged.
It does not repair or qualify the larger learning graph.
[Read the isolated-bridge assessment](../studies/20261023-fixed-bridge/ASSESSMENT.md).
Recount it with `python -m resident.bridge_audit`.

Measured in this campaign: 6 owned sessions, 663 actual forwards and 796 complete
per-bank state captures. The matched repair/frozen/wrong-bank application stage
was never admitted. No new useful-task, CPU or energy advantage is established.

![Native memory checks stop the mapped scale-up](../visual/resident-boundary-4k.png)

* [Full assessment](../studies/20261023-resident-push/ASSESSMENT.md)
* [Frozen task and stop rules](../studies/20261023-resident-push/PROTOCOL.md)
* [Next isolated contract gates](NEXT.md)
* [Numeric archive](../results/2026-10-23-resident-boundary.json)

Hardware-free verification, requiring NumPy:

```bash
python -m resident.archive audit-public
python -m unittest discover -s tests -p 'test_resident_boundary.py' -v
```

`design.py` contains the synthetic task and exact-rule reference. `prepare.py`
contains the **failed final adapter**, not a qualified native producer; it needs
a separately licensed Akida SDK. `partition.py` preserves the rejected image
partition design. `src/resident_bench.cpp` is the observed-state instrument.
`run.py` requires a new bounded authority, source freeze, stage admission and
exclusive device lock; the saved campaign is closed. Do not reopen its state.

`archive.py` checks all successful prefixes and the exact first failure in every
failed attempt, including missing-output and partial-snapshot boundaries. The
public package excludes proprietary programs/runtime sources. Optional figure
regeneration requires Matplotlib: `python -m resident.report`.
