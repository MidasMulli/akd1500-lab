# AKD1500 Lab

Reproducible native-learning checks, matched CPU benchmarks, mesh-resource
probes, and external power-recorder tools for BrainChip's AKD1500.

This is an independent research toolkit. It measures what a mapped program
actually does and preserves the distinction between mapping, correct execution,
performance, and electrical energy. It does not establish a neuromorphic
advantage by assuming one.

## First measured study — October 23, 2026

**22 passing physical sessions and two preserved failed burst attempts on one
chip.** The failures exposed a withheld full-ring queue-publication problem;
leaving a spare descriptor slot restored repeated-burst execution.

The 32-NPU mesh passed twice. Its evidence run contains 64 observations,
4,096 exact output scores and 64 per-step learning-RAM readbacks. In that run,
54 of 64 learner records changed, with 1,004 connection replacements.
This is routing/learning integrity, not peak utilization or a useful CNN task.

| Kernel | CPU reference, µs/input | Native + host interface, µs/input |
|---|---:|---:|
| 128 inputs / 8 neurons | 0.46 | 43.4 |
| 512 inputs / 64 neurons | 3.39 | 152.6 |
| 512 inputs / 512 neurons | 16.53 | 961.6 |

Medians of three paired run means, each with 256 native examples and 256 CPU
repetitions of the same stream. These include native input/submission/completion/
output costs. They establish **no speed or energy advantage for the AKD** on
these workloads. Clock governor remained dynamic; this is a system-path result.
Output drain accounted for approximately 94% of the larger resident-burst
transaction. That separation is not a measurement of core-only execution time.

![Measured comparisons](visual/measurement-results-4k.png)

[Public results](results/2026-10-23.json), [raw mesh evidence](results/mesh32-evidence.npz),
[4K60 recorded replay](visual/mesh32-replay-4k60.mp4),
[remaining hardware gaps](ROADMAP.md). **No calibrated card-energy result yet.**

## What is here

* A bounded PCIe MMIO adapter using a separately supplied Akida runtime.
* A C++ CPU bitset learner and an independent Python integer/set verifier.
* Exact per-output and complete learning-RAM comparisons, not classifier-only checks.
* Input-width/neuron-count compiler probes that retain unsupported mappings.
* A fixed-permutation mesh ladder ending in native association learning,
  including a compiler-generated single-pass 32-NPU configuration.
* Input/submission/fetch timing and an experimental resident-burst instrument.
* INA228 recorder firmware, raw capture, calibration gates and energy analysis.
* Data-bound figures and a replay of logical placements and measured learning.

Read [the measurement protocol](studies/20261023-measurement/PROTOCOL.md),
[mesh contract](studies/20261023-measurement/MESH_PROTOCOL.md),
[resident timing boundary](studies/20261023-measurement/RESIDENT_PROTOCOL.md),
and [power-rig build plan](power/BUILD.md) before interpreting results.

## Dependencies and reproduction

Tested on Linux AArch64, Jetson Orin Nano, PCIe AKD1500 and Akida 2.19.3.
BrainChip's licensed runtime sources, SDK headers, generated program images and
device firmware are **not included**. Obtain them through the vendor's supported
distribution. No installed driver, SDK, clock or voltage settings are changed.

```bash
python -m unittest discover -s tests -v

# On the Linux machine physically hosting the card:
export AKIDA_ENGINE_ROOT=/path/to/your/runtime
export FLATBUFFERS_INCLUDE=/path/to/compatible/flatbuffers/include
bash tools/build.sh

python tools/new_study.py --name my-run --minutes 30
export AKD_LAB_STUDY=studies/my-run
python tools/prepare.py "$AKD_LAB_STUDY/fixtures"

# Start with a small, correctness-gated cell.
python tools/run_cell.py smoke "$AKD_LAB_STUDY/fixtures/n128_u8" \
  --samples 16 --cpu-repeats 128
python tools/verify.py "$AKD_LAB_STUDY"

# Compiler-only resource probe; inspect before admitting hardware work.
python tools/prepare_mesh.py "$AKD_LAB_STUDY/mesh"
```

The local study envelope enforces an expiry, session/forward caps, a shared
`/tmp/akd1500-research.lock`, occupied-device refusal and process cleanup.
Start a fresh explicitly invoked study when an envelope expires. These are
finite research tools, not unattended services. CPU affinity currently targets
core 2; inspect suitability for another host before running.

## Evidence boundaries

Kernel timing includes input MMIO, submission, completion polling and output
readback. Compilation, programming, reference calculations and RAM snapshots
are excluded and described separately. Busy polling is host work. Full score
readout can dominate a small PCIe workload; it is not intrinsic silicon latency.

The CPU uses population counts and the reconstructed binary learning rule,
not a deliberately slow dense neural simulator. It is an optimized reference,
not a claim of the best possible CPU implementation. The mesh uses fixed channel
permutations composed algebraically for its oracle; **that family establishes
routing integrity, not useful CNN accuracy or a hardware speed advantage**.

All 32 mapped NPUs does not mean 100% utilization, maximum memory capacity,
32 independent learners or a chip-wide learned communications protocol.
Logical placement is not a physical die floorplan. Intermediate visual motion
is explanatory replay, not electrical spike or current measurement.

The FNP3 decoder currently excludes FNP2 external-memory records. A compiler
mapping that exceeds the decoder is a tooling gap, not a measured chip limit.
Firmware, hidden learning state, restart durability, sparsity scaling, complete
FNP2 support and calibrated card/host energy remain explicit investigation areas.

The power recorder has synthetic software tests. Physical energy claims require
an assembled, calibrated sensor and complete interception of the measured supply
branch. Module power, card power and silicon-only power are different domains.

## Public evidence and development

Compact verified summaries are published under `results/`. Full raw local runs
include score vectors, learning-RAM words, timing samples, source/program hashes,
failed attempts and cleanup receipts. Local raw data and operator authorizations
are excluded from Git by default. The exporter sanitizes private paths.

Do not silently discard failed runs, call a compiler rejection a physical limit,
or claim a 10x advantage using mismatched models, accuracy, update budgets,
memory capacity, duty cycles or power boundaries.

## Sources

[Akida runtime API](https://doc.brainchipinc.com/api_reference/akida_apis.html),
[Akida engine integration](https://doc.brainchipinc.com/user_guide/engine.html),
[TI INA228 datasheet](https://www.ti.com/lit/ds/symlink/ina228.pdf),
[NVIDIA module power monitoring](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).

Own tooling is MIT licensed. Vendor dependencies retain their separate licenses.
