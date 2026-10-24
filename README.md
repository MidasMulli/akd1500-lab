# AKD1500 Lab

Reproducible native-learning checks, matched CPU benchmarks, mesh-resource
probes, and external power-recorder tools for BrainChip's AKD1500.

This is an independent research toolkit. It measures what a mapped program
actually does and preserves the distinction between mapping, correct execution,
performance, and electrical energy. It does not establish a neuromorphic
advantage by assuming one.

## Native execution and causal NPU checks — October 23, 2026

**28 physical sessions, 9,824 forwards, exact numerical checks.** A hardware
counter study separates neuron-population scaling, activation sensitivity,
multi-NPU partitioning and teaching queue behavior. A separate causal ablation
reads actual weight/threshold/readout memory from all three NPUs and suppresses
each compute group's thresholds in turn: only that group's output disappears.

At half-active inputs, 128 fixed convolution outputs on one compute NPU use
16,213 median DMA-monitor cycles; 256 outputs on two use 16,223.5. In the tested
FNP3 path, more neuron records instead add approximately 20 monitor cycles each.
Eight-input teaching bursts reproduce exact outputs and RAM with both the
normal setting and a bounded direct pipeline-flag test. These distinguish
resource behavior; they do not establish per-neuron clock timing or a CPU/energy
advantage.

![Native execution measurements](visual/native-execution-4k.png)

[Methods and open questions](metal/README.md) ·
[execution assessment](studies/20261023-metal-characterization/ASSESSMENT.md) ·
[causal ablation](studies/20261023-metal-ablation/ASSESSMENT.md) ·
[public recount](metal/audit_public.py)

## Two resident learning memories, selectively repaired — October 23, 2026

**Recognition can adapt while a connected routing memory keeps its rule.**
This synthetic sensor-routing task uses five native NPUs: two independently
taught 64-neuron learning banks, a fixed native comparator and categorical
encoder between them, and a fixed four-count output. All intermediate neural
processing stays in one resident hardware pass.

After beacon signatures change, recognition-only repair recovers **64/64**
correct destinations while preserving **64/64** old-signature responses.
Routing's complete captured learning RAM stays unchanged. Frozen and routing-
only repair each reach **16/64** on changed signatures; routing-only repair
also reduces old-signature performance to **48/64**. All three initializations
reproduce these outcomes on one chip and one synthetic sensor world.

![Selective native recognition repair](visual/signal-relay-4k.png)

[Download 4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/signal-relay-4k60.mp4),
[tools and explanation](relay/README.md),
[assessment](studies/20261023-signal-relay/ASSESSMENT.md),
[results](results/2026-10-23-signal-relay.json).

Every one of **9,280 native forwards** and **18,560 complete learning-bank
snapshots** passes independent reconstruction. The host supplies different
teachers and schedules which bank learns; this is not autonomous teacher
selection, simultaneous independent streams or recurrent planning. A simple
CPU comparator also solves the task. **No CPU speed or energy advantage is
established.** The 24-second film is a time-expanded recorded replay: RAM edits
and final counts are measured; intermediate responses and flow are reconstructed
or schematic. Sources and numerical archives are public; generated vendor
programs and runtime remain external.

## Native relearning and a direct CPU cost comparison — October 23, 2026

**The current Pong pipeline does not establish a neuromorphic advantage.**
After identical new-rule teaching, an AKD learner without old court-A labels
scores **46/64**, an old-A learner adapting to the new rule scores **28/64**,
and trained-then-frozen scores **12/64**. The practical CPU recent-neighbour
learner scores **44/64**. All three native arms retain **47/64** on court B
without further B teaching. This is one physical initialization; no native
accuracy advantage over CPU is established by the two-example difference.

![Native learning-history diagnosis](visual/pong-relearning-4k.png)

The fresh A arm fits its repeated teaching cases at 63/64, while relearning
reaches 54/64. Prior learning history impairs this case; the physical cause is
not uniquely isolated. Both prospectively fixed connection-count repairs fail
their exact-rule CPU screens, so native seed expansion, confirmation and paddle
control stay closed. Forecasts are labeled separately from native results.

**The same recorded neural task is also cheaper on the CPU in every tested
transport configuration.** Exact CPU replay uses about **11.2 µs/input**.
Serial MMIO costs **342.6 µs/input** and **342.3 µs/input** of caller CPU.
A targeted queued read-only DMA path reduces caller CPU to **54.7 µs/input**
and elapsed cost to **256.0 µs/input**, averaged over three repeats. This is an
84% reduction in caller CPU versus our MMIO path, while exact CPU replay still
uses less. Queued elapsed/input is batch-amortized cost, not event latency.

![Matched recorded-Pong transport and CPU cost](visual/pong-substrate-4k.png)

[Tools and reproduction](pong/relearn/README.md),
[learning assessment](studies/20261023-pong-relearning/ASSESSMENT.md),
[primary cost assessment](studies/20261023-pong-substrate/ASSESSMENT.md),
[read-only DMA follow-up](studies/20261023-pong-read-dma/ASSESSMENT.md).
The primary four modes were frozen before execution; the read-only follow-up
was separately frozen after reviewing their input/submit/fetch breakdown.
All **47,072 new native forwards** pass independent checks: 6,752 have complete
per-step RAM captures; the 40,320 transport replays have every score and full
final RAM bound to the original physical stream. No installed driver or runtime
was changed. These are current program/host-path costs, not silicon-only or
electrical energy measurements, and do not establish chip-peak capability.

## Changing-physics Pong with a trained-then-frozen control — October 23, 2026

**Native learning responds to the changed rule, but useful prediction remains
unqualified.** Both arms first replayed the same 512 labeled presentations and
reached identical complete trained RAM and scores. After court A's physics
changed, the continuing learner predicted the correct landing band on **12/48**
separate development trajectories; the trained-then-frozen arm scored **4/48**.
The evaluation-only best constant scored **19/48**. The preset 65% accuracy gate
failed, so additional seeds, confirmation trajectories and learned paddle
control stayed closed.

![Changing-physics Pong and measured native predictions](visual/pong-physics-4k.png)

[Download 4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/pong-physics-4k60.mp4),
[results](results/2026-10-23-pong-physics.json),
[assessment](studies/20261023-pong-physics/ASSESSMENT.md),
[tools and reproduction](pong/README.md).

The 23-second film captures 16 wall-paced serves per arm, sequentially aligned
at their starts. Paddles remain fixed; the colored landing marker is the actual
native prediction. Teaching follows the observed landing. All 160 integer
match values and actual connection edits are traceable to the public raw
archive: **4,140 physical forwards and complete per-forward RAM checks**.
The filmed learner made 174 native connection replacements; its frozen control
made zero after their shared training. The host supplies the simulator,
position/velocity encoding and four-band decoder. No pixel-learning, electrical
spike, full-chip, energy or useful Pong-control claim follows from this result.

## Continuous Mario decisions with matched frozen controls — October 23, 2026

**The native loop now chooses again every 24 game frames and learns from each
executed action. Its first matched pair regressed, so expansion stopped.**
Across four development starts, the learned arm advanced 1,418 pixels versus
frozen 1,554. Frozen suppresses both preparation and online learning; this
comparison does not isolate online updates after shared training. All eight
episodes ended in death; neither arm completed a level.
The predeclared regression gate closed new seeds and left confirmation starts
unopened. This is a functioning continuous interface, not a learning advantage.

One resident bank made 46 learned and 69 frozen decisions, with **507 native
connection replacements online** in the learned arm and zero RAM changes in
frozen. All 3,385 forwards, complete per-step RAM, rendered frames, input
features and action/outcome bindings passed independent verification. Game
resets occur only between labeled episodes; neural state stays resident across
the four episodes. Short actions, RGB/time features, ranking and teaching labels
are host supplied. The chip learns associations natively.

![Continuous matched hardware recordings](visual/mario-continuous-4k.png)

[Download paired 4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/mario-continuous-4k60.mp4),
[results](results/2026-10-23-mario-continuous.json),
[assessment](studies/20261023-mario-continuous/ASSESSMENT.md),
[tools and reproduction](mario/continuous/README.md).
The separate physical runs are aligned at each episode start. A darkened view
means that arm died while its counterpart continues; later-episode responses
are never displayed on the ended game. Cell brightness is a measured match
response, not an electrical spike trace. Training had only 4 unsafe outcomes
out of 192, limiting hazard coverage. A new data/representation development
stage is needed before another qualification attempt.

## Larger sealed Mario test and live chip capture — October 23, 2026

**Learning did not pass the larger confirmation gate.** Five matched physical
learned/frozen pairs on 128 prospectively fixed encounter timings scored
**551/640 learned versus 534/640 frozen**. Learning won four pairs, but the
2.66 percentage-point gain missed the preset 5-point threshold. On the 25
contexts where the action matters, the conditional gain was 13.6 points and
its block-bootstrap 95% interval was **−2.86 to +25.6 points**, including zero.
The best constant action scored **555/640**. These are repeated initializations
on the same bank and one chip, not 640 independent environments.

Every native response and captured RAM state passed the independent Python
reconstruction. All 1,280 learned/frozen selected-action outcomes also matched
fresh emulator executions. Earlier failed development gates remain failed;
this separate confirmation was registered before collecting its test outcomes.
No representation or training change was made in response to those outcomes.

**The video contains actual hardware responses alongside wall-paced gameplay.**
It preserves the measured timing: about 56 captured game frames/s and 8.4 pairs
of chip queries/s. It shows 45 native connection replacements after four actual
outcomes. Those clips use selected development cases; the table shows the
separate sealed confirmation. Colored cells display measured match responses,
not electrical spike traces. The host still supplies RGB/time features, ranking
and two fixed 90-frame actions; shadow queries do not replan the action.

![Synchronized gameplay and native responses](visual/mario-live-responses-4k.png)

[Download 4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/mario-live-responses-4k60.mp4),
[confirmation results](results/2026-10-23-mario-confirmation.json),
[protocol and assessment](studies/20261023-mario-confirmation/ASSESSMENT.md),
[tools and reproduction](mario/sealed/README.md).
The public archives include all 25,406 native forwards from this extension,
their inputs, labels, outputs and per-step RAM; no ROM or vendor runtime.
This establishes working native learning and instrumentation, while a reliable
Mario advantage and CPU/energy advantage remain unproven.

## Recovery and a first Mario skill screen — October 23, 2026

**The image service now tests whether a failed route has recovered.** A supplied,
bounded probe clock requests exploration; the native circuit chooses the action
and updates its consequence memory from the actual worker result. In three fixed
encodings on one chip, it corrected stale decoder-failure memories by requests
13, 14 and 15. The reactive control made no recovery query during that window.
Both completed 72/72 PNGs; probing cost 156 operations versus 154. Frozen
transition-memory controls completed 53/72. Earlier knowledge is demonstrated;
more completions than the reactive control and a substrate advantage are not.

All native decisions, outputs and captured learning RAM passed independent raw
recounts. The identical CPU algorithm reproduced every action and outcome.
The missing-FlatBuffers build failure and an unaligned-access transport failure
remain preserved. The repair used a fresh namespace and left neural programs
byte-identical. No installed driver, runtime or firmware was changed.

![Measured recovery replay](visual/recovery-replay-4k.png)

[4K60 recovery replay](visual/recovery-replay-4k60.mp4),
[results](results/2026-10-23-recovery.json),
[protocol and assessment](studies/20261023-recovery-r1/ASSESSMENT.md),
[reproduction](image_service/README.md). Link slots and placement are schematic;
connection contents come from measured RAM. The example JPEG is our test fixture.

**The first Mario test is mixed.** A 96-record native association bank received
authored binary features from two rendered frames plus a candidate action.
After four fixed epochs of 12 labeled experiences, each initialization selected successful
actions at 7/8 nearby held-out starts. Frozen controls scored 6/8, 8/8 and 6/8;
action-only controls scored 5/8 each. Learned totals were 21/24 versus frozen
20/24 and best constant 18/24. The predeclared two-of-three screen passed, but
this tiny aggregate gain and the stronger frozen seed prevent a robust learning
claim. All 72 selected-action trials matched in fresh emulator episodes.

The host supplies the visual encoder, ranking and two 90-frame motor primitives.
The chip changes selected binary connections natively. This is recorded-experience
consequence learning followed by fresh action execution, not full-game RL,
learned perception, frame-by-frame native control or transfer to unseen levels.
Every score and full RAM snapshot matched the C++ and independent Python models;
queries left visible RAM unchanged. Hidden-selector or power-loss durability
does not follow.

![Mario pilot including a harmful update](visual/mario-consequence-4k.png)

[4K60 comparison](visual/mario-consequence-4k60.mp4),
[results and limitations](results/2026-10-23-mario.json),
[public raw scores/RAM](results/2026-10-23-mario.npz),
[fresh episode receipts](results/2026-10-23-mario-fresh.json),
[protocol and reproduction](mario/README.md).
Hardware-free tests reconstruct all 752 native inputs and reject forged matching
CPU/native scores and query-state edits. The next useful gate is a larger,
prospectively sealed encounter bank with enough independent variation to resolve
the frozen-control result; do not tune this test after seeing its outcomes.

## Connected learning and host offload — October 23, 2026

**Three connected native learning banks passed exact physical checks.** Each
has 64 configured neurons, 512 binary inputs and 32 selected connections per
neuron. Two fixed encoders and a fixed two-count readout bring the graph to six
FNP3 placements. Every stage receives the same external class label; updates
are local native feature/connection selection, not backpropagation.

We checked the full RAM of every bank after every input, final output counts,
and fixed encoder/readout memory. Three seeds passed. Freezing the middle bank
kept its complete RAM fixed while both neighbors continued learning; freezing
all banks prevented every connection edit. Intermediate neuron responses in
the replay are reconstructed from the exact reference; RAM edits and final
counts are measured.

A first positive-only fanout saturated all 64 final-bank responses in 135 of
192 confirmation inputs. A fixed balanced code—four response copies and four
complement copies—reduced that to 0/192 on the same seeds and input streams.
This repairs a representation problem; it is not task-generalization evidence.

The proposed larger CNP-interleaved family failed physical reference checks,
including its untouched compiler inference graph. We stopped that family and
retained all ten failed/diagnostic attempts, including one invalid pre-device
invocation and three host cleanup crashes. The cleanup buffer-lifetime bug was
fixed. No four- or seven-bank physical success is claimed.

![Recorded three-bank state replay](visual/network-replay-4k.png)

[4K60 recorded replay](visual/network-replay-4k60.mp4),
[balanced-bank results](results/2026-10-23-balanced-network.json),
[per-input RAM and outputs](results/2026-10-23-balanced-network.npz),
[earlier branch and preserved failures](results/2026-10-23-network.json),
[balanced protocol](studies/20261023-balanced-network/PROTOCOL.md).
The routes, neuron positions and motion are schematic, with time expanded.

The host-offload experiment also passed matched-output/RAM checks:

| Same output within each pair | Host transaction, µs/input | Calling-thread CPU, µs/input |
|---|---:|---:|
| 512 scores / MMIO | 961.48 | 960.43 |
| 512 scores / kernel DMA read | 951.65 | 104.56 |
| Two counts / queued busy polling | 69.25 | 69.14 |
| Two counts / queued 50 µs sleeps | 79.79 | 29.12 |

Medians of three interleaved run means. DMA reduced calling-thread CPU about
9.2 times at similar full-score latency. Queued sleeps reduced caller CPU
about 2.4 times, with about 15% more latency. The initial small DMA smoke was
much slower and remains preserved; its cause is unresolved. Caller CPU is not
whole-system CPU/IRQ accounting. **No electrical-energy advantage is established.**
Two sparse-input metadata adapters timed out and are preserved. The available
driver source has DMA-transfer completion interrupts; this does not establish
an exposed inference-completion interrupt. Exact loaded-module source identity
is unverified. No installed driver/SDK was changed.

![Host-offload measurements](visual/offload-results-4k.png)

[Offload results and failures](results/2026-10-23-offload.json),
[raw outputs, RAM and timing](results/2026-10-23-offload.npz),
[protocol](studies/20261023-host-offload/PROTOCOL.md).

The shorter wait was separately applied to the existing **two-memory image
service**, which processes pixels on the Nano and sends compact job outcomes
to the AKD. Both wait paths completed 24/24 real image jobs through owned worker
outages, with identical actions/outcomes and every neural-memory snapshot.
The CPU controller matched every decision/outcome too. Removing the previous
deliberate 20 ms guard is an interface change, not a silicon-compute speedup.
The new three-bank synthetic cascade is not yet this application controller.
[Image-service evidence](results/2026-10-23-image-offload.json),
[packaged image-service tooling](image_service/README.md).

Hardware-free tests reconstruct all published passing offload and network
streams and reject corrupted output/RAM even when their file hashes are
updated. The remaining electrical gate is an assembled, calibrated direct
card-power rig; module rail readings cannot replace it.

## Compact-output follow-up — October 23, 2026

**28 additional physical sessions passed, covering 5,056 native inputs.** The
first stage compared full potentials with threshold responses. After exact
verification, the second added a fixed downstream NPU that sums each of two
groups of 256 threshold responses. The 512-neuron bank still learns natively.

| Output contract | Payload bytes/input | Matched CPU, µs/input | Native + host, µs/input |
|---|---:|---:|---:|
| 512 integer match scores | 2,048 | 16.52 | 961.88 |
| 512 responses above threshold seven | 512 | 16.65 | 287.64 |
| Two group-support counts | 8 | 17.56 | 73.70 |

These are medians of three paired run means. Returning two counts reduced the
measured native host transaction time about 13 times relative to full scores.
**The contracts carry different information:** thresholds discard magnitudes;
counts also discard neuron identity. This is not a speedup at the full-score
contract, nor a classification-accuracy result. One-bit dense responses occupy
one byte per neuron on this transport. Payload excludes headers, input traffic,
descriptors and polling.

Every response/count and the full learning state matched an independent Python
selector model. All final learned connection/adaptive-threshold words agree across
contracts; the mask and downstream-reducer versions also have exactly identical
complete final learner RAM (10,242 words). The fixed downstream sum is programmed by
us. This two-NPU experiment is separate from the 32-NPU routing-coverage study.

Batching eight jobs did not materially improve the mask transaction in this
screen. An inference-only early-exit CPU variant was also tested; it did not
beat the regular CPU control in that single probe. The CPU remains faster in
the confirmed matched learning comparisons. **No electrical efficiency
advantage has been established.** Host busy polling remains part of the path.

![Compact-output measurements](visual/readout-results-4k.png)

[Verified results](results/2026-10-23-readout.json),
[raw outputs, learning RAM and timings](results/2026-10-23-readout.npz),
[4K60 paired recorded replay](visual/readout-replay-4k60.mp4),
[threshold-output protocol](studies/20261023-compact-readout/PROTOCOL.md),
[downstream-reducer protocol](studies/20261023-class-reduction/PROTOCOL.md).
The replay pairs separate matched native executions; it is not live wire tracing.

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
* Matched potential, threshold-mask and two-count output contracts with a
  guarded intermediate-learning program assembler and independent verifiers.
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

For the compact-output instrument, build with `AKD_LAB_BENCH=readout_bench` and
generate fixtures using `tools/prepare_readout.py`. Pass
`--bench readout_bench` to `tools/run_cell.py`. Start with a 16-sample potential
and mask pair, then use `tools/verify_readout_pair.py` before expansion.
`tools/prepare_reduction.py` accepts the reviewed mask-fixture directory and a
new output directory; `tools/verify_reduction_pair.py` checks its native counts
and complete learned RAM against the matched direct-mask stream. The assembler
supports only the reviewed SDK 2.19.3 single-pass FNP3 format and rejects other
tracks or unfamiliar register differences. It does not change installed SDK,
driver or firmware files. `program_codec.py` is reused from the earlier native
image-service tooling; it is our adapter, not vendored runtime implementation.

Hardware-free CI reconstructs and independently verifies all 28 published
follow-up streams from the raw archive, and rejects deliberately corrupted
counts even when the native and C++ reference files agree with each other.
The earlier preserved failures remain in the first study's results.

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
