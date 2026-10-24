# Native AKD operations: physical evidence and a CPU control

We separate three questions: what the device executes natively, how its resource
cost scales, and whether the implementation beats a CPU under useful constraints.
The current study advances the first two. **The optimized CPU is faster on the
tested fixed binary workload; no calibrated energy advantage is established.**

![Measured native execution and CPU controls](../visual/native-primitives-4k.png)

[4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/native-primitives-4k60.mp4) ·
[primary assessment](../studies/20261024-native-primitives/ASSESSMENT.md) ·
[11-region causal checks](../studies/20261024-cnp-contribution/ASSESSMENT.md) ·
[runtime allocation diagnosis](../studies/20261024-queue-allocation/ASSESSMENT.md) ·
[remaining gates](NEXT.md)

## Native operations demonstrated

| Operation | What physical measurements establish | What remains separate |
|---|---|---|
|Fixed low-bit neural computation|Identical binary overlap/threshold/group-count mathematics runs on FNP and CNP graphs; all outputs and parameter regions check|Exact internal arithmetic circuit, transistor layout and general chip peak|
|Distributed CNP contribution|A 2,048-detector graph uses 11 compute CNP regions plus a readout; silencing each region's outputs causes the exact predicted change|Physical power gating and literal same-clock overlap|
|Resident learning and communication|Three plastic populations communicate through fixed encoders; recognition can change connections while downstream memories remain fixed|Arbitrary update-algorithm upload, independent teacher streams and recurrent autonomy|
|Learning outside the host worker|32 queued updates complete while that thread blocks, with no driver polling; native outputs and captured RAM match the polling control|All-host sleep, live sensor autonomy, interrupt completion and electrical energy|

These native device operations need a different implementation on the CPU. That
fact is compatible with CPU emulation being faster for a particular small task.
The CPU controls use packed NEON, an inverted connection index and a fixed
adaptive policy, not a scalar Python loop. They receive the same dense bytes and
produce the same two counts; packing is included and resident setup excluded.

Increasing CNP detectors from 128 to 2,048 grows the population **16-fold** while
half-active DMA buffer service grows **1.39-fold**, reproduced across three temporal
sessions. This measures aggregate resource behavior, not per-neuron clock traces.
At 2,048 detectors the packed CPU uses about **9.20 µs/input**, while the measured
native CNP serial host path uses **86.96 µs/input**. The latter includes host I/O;
it is not a silicon-only latency measurement or proof no faster native path exists.

## Preserved boundary

The 128-event learning stage requires 129 held-ring slots. The runtime returns
128; the guard fails before repair and the actor then crashes during exception
cleanup. The full 512-event prefix independently checks. A separate zero-forward
diagnostic confirms the allocation clamp and cleanly unprograms the owned device.
The failed learning stage is not retried or reported as a success.

The published `src/quiet_relay.cpp` is hardened **after closeout**: program storage
outlives cleanup and the unqualified 128-event case is refused before device
access. Exact executed v1 is preserved under `executed/quiet_relay_v1.cpp` for
receipt verification, not recommended for execution. The lifetime hazard is
identified; its responsibility for the precise crash signal is not uniquely
isolated. Source hashes distinguish executed and hardened versions.

Across all three closed studies: **43 owned device sessions, 9,040 verified
forwards and 7,404 complete learning-bank RAM captures**. The zero-forward
allocation session and failed prefix remain explicit. Fixed graphs have three
temporal repeats; quiet-learning arms have one initialization and one run each.

## Recount without hardware

NumPy suffices for numerical verification:

```bash
python -m capability.archive audit-public
python -m capability.contribution_archive audit-public
python -m unittest discover -s tests -p 'test_native_primitives.py' -v
```

Every captured learning-RAM boundary is retained losslessly as changed-word
deltas plus each original full snapshot's SHA256. Intermediate states inside a
queued chunk are not presented as measured. Fixed parameter packets, dense inputs,
outputs, timing samples and CPU results are numeric public data. Vendor runtime,
generated programs and operator authorizations are external/private.

`design.py` freezes the mathematics and input/weight seeds. `prepare.py` and
`contribution_prepare.py` require the separately licensed Akida compiler but
never discover a physical device. `run.py`, `allocation.py` and
`contribution_run.py` enforce deadlines, source identity, staged gates, exclusive
ownership and raw receipts. `audit.py`, `failure.py` and both archive modules
independently reconstruct outcomes. Actors live under `src/`.

Physical reproduction needs a **new** bounded authority, source/binary/fixture
freeze and staged admission. The saved campaigns remain closed. No installed
driver, runtime, firmware or clock was modified. Native sources require the
separately supplied compatible runtime; the CPU primitive is standalone C++.

## Presentation

Matplotlib, Pillow and FFmpeg add scientific plots and a recorded evidence replay:

```bash
python -m capability.report
python -m capability.visual --video
```

The video keeps the fixed and learning graphs in separate panels. NPU positions
are a logical resource map; highlighted suppression is a parameter intervention.
Learning-record spokes represent actual stored input addresses, with schematic
angles. There is no electrical spike trace or invented completion timestamp.
Its source binding includes both numerical archives, derived metrics, replay
data, presentation sources, stills and the MP4. Geometry and playback time do not
imply simultaneous execution of the two different experimental graphs.
