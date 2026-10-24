# Input-gated native learning and a paced GPU comparison

**Native input-driven eligibility passed. A useful CPU/GPU resource advantage did not.** The closed campaign contains 22 completed physical sessions and 20,998 AKD forwards, plus one preserved non-executed lock admission. No owned job remains running. The original X draft was saved earlier, not published; the new film is separate review material.

## What changed

The primary AKD arm keeps native learning enabled and uses the same class value, 0, for every event. A fixed sparse input encoding and learning threshold distinguish prediction probes from actual outcomes. The host does not select a winning record, upload new connection weights, or switch learning mode between events in that arm. This is a newly qualified design in our toolkit, not a claim of a previously unknown silicon capability.

Each of 16 local-SRAM FNP3 records starts with 10 context anchors, 16 feedback anchors and 6 outcome connections, for 32 connections total. A prediction probe contains only the context and a candidate outcome: at most 16 matches, below learning threshold 24. Actual outcome feedback adds the 16 feedback anchors. The correct context can then match at least 26 connections; other contexts reach at most 22. The native rule can replace outcome connections while preserving the active context/feedback anchors. Every query's visible connection memory stays unchanged even though learning remains enabled. Hidden selector state is not claimed to remain unchanged.

A second, fixed FNP3 reduces activated learner records to a support count. Two sequential candidate probes produce a prediction. The fixed readout is a programmed reduction, not a second learner or autonomous planner. Raw potentials and reduced readouts come from paired recordings whose connection/learning trajectories match exactly; output/routing auxiliary words legitimately differ. Direct SRAM readback confirms the local record packet, and fixed readout memory stays unchanged.

The host still supplies events, fixed encoding, constant class fields, queue descriptors and initial configuration. Four-bit temporal history is encoded outside the chip. Feedback anchors are an explicit external observation marker. No internal prediction-error generator, indefinitely recurrent scheduler, autonomous input source, power-loss persistence or whole-chip plasticity was demonstrated.

## Sealed task and controls

Three fresh seeds each learn 16 binary transition rules, then two rules change. A trained-then-frozen control and a disconnected-feedback control share the entire initial learning prefix. The disconnected arm keeps learning enabled but omits feedback anchors after the prefix. Neither control repairs the changed rules. All contexts have been taught before the sealed trajectory tests; new trajectories are not new-context generalization.

- Initial trained baseline: **48/48** predictions, in every arm.
- Changed-rule post-adaptation check: **6/6 adaptive, 0/6 frozen, 0/6 disconnected**.
- Unchanged-rule retention check: **42/42 in every arm**.
- New temporal trajectories: **576/576 adaptive; 511/576 frozen and disconnected**. The 65 failures in each control all occur on changed contexts; unchanged contexts remain correct.
- Every native output, full per-event SRAM snapshot and fixed readout-memory check matches independent Python and exact C++ replay. Native and disconnected arms use no per-event mode switching. The frozen arm deliberately disables learning once after the shared prefix; diagnostic readbacks temporarily enable access without a following forward.
- A separate optimized CPU table learns the same task successfully. Exact native-rule CPU execution is about **2.6 µs/event**, and the optimized table about **1.1 µs/event**, including timing overhead. Serial native readout is about **48 µs/event** here. This task is cache-friendly and does not establish a speed advantage.

Three queued runs process 32 events at a time while the host worker sleeps 5 ms. There are no driver reads during each blocked interval and only the release write. All queued outputs are ready at the first wake, and outputs and complete memory match sequential execution at every boundary. This is bounded queued execution with externally prepared events, not self-sustaining native recurrence. The 5 ms interval is an observation bound, not the device's measured completion latency.

## GPU co-running result

The unchanged T202 Nano world model runs a fixed valid input/action forecast service. The port manifest and checkpoint are hash-checked; every returned RGB output is identical, and model weights remain unchanged. GPU service is pinned to cores 0–1; the background actor to core 2. Twelve runs use three interleaved blocks: GPU alone, GPU plus exact CPU learner, GPU plus task CPU, and GPU plus AKD. Each background actor processes 3,072 identical events at 600 events/s over 5.12 s, with persistent state across four streams and the same scheduled wake times. No unpaced CPU stressor is used.

| Background service | Median GPU forecasts/s across 3 runs | Median actor CPU time per 5.12 s |
|---|---:|---:|
| None | 27.69 | — |
| Exact CPU learning rule | 27.61 | 6.33 ms |
| Optimized task CPU | 27.74 | 1.33 ms |
| Native AKD path | 27.66 | 78.67 ms |

All native/CPU prediction results and continuous final SRAM independently replay. The optimized CPU's prediction probes agree with the native task outputs; feedback acknowledgments have different meanings and are not compared as predictions. One optimized-CPU batch finished more than 2 ms after its scheduled wake; preserve it as a timing exception. The independent audit therefore reports a continuous-replay pass **with a deadline exception**, not universal deadline success. It was not discarded or rerun.

The third native GPU run is slower than the first two; three temporal blocks and changing system conditions do not identify a cause. The co-run does not show a reproducible GPU-service benefit, and native host CPU use is higher. This small model fits CPU cache easily; these results do not disprove a bandwidth advantage for a larger DRAM-bound workload. Tegrastats records controller load, clocks and temperatures but cannot assign per-client RAM traffic. No DRAM-contention cause, calibrated energy advantage or broad CPU/AKD superiority is established.

The existing PyTorch 2.14.1+cu130 environment warns about Orin CC8.7 support. The tested forecast operations complete with valid, nonconstant, identical outputs. The warning and numeric profile are retained; no package, model, firmware or clock was changed. This qualifies the observed workload, not general runtime support.

## Visual and reproducibility

The 24-second 4K60 and 1080p60 films show: an old rule responding, a changed rule initially missing, an actual outcome crossing the learning threshold, six connection replacements, the corrected response, and an unchanged rule retained. Numeric native responses and connection masks are measured; frozen potentials are reconstructed from frozen SRAM. Geometry, camera and expanded playback time are schematic. No electrical spike timing is invented. Every video decodes to 1,440 distinct frames and has a fast-start MP4 index.

```sh
python -m feedback.archive audit
python -m feedback.corun_audit audit
python -m feedback.visual --out /absolute/output --video
```

Numeric archives contain own model arrays, results, hashes, controls and failed attempts. Vendor runtime/headers and generated programs are not redistributed. The source/admission freeze and GPU addendum are distinct, prospective records.

## Failures and next boundary

Compiler preflight 01 incorrectly required raw-potential and activated-output packets to have identical auxiliary words. That assertion failed before hardware; preflight 02 separately preserves legitimate output auxiliaries and proves selected connections and learning parameters identical. Build 01 failed on a mixed-type C++ auto declaration and was repaired before hardware. One attempted admission encountered our own already-held device lock; no device was opened or child launched. Its directory and ownership receipt are preserved and conservatively counted against the budget before the sequential retry. No physical correctness or memory mismatch was observed.

A candidate follow-up must remove more external control or establish a resource benefit. The specific autonomy gap is native temporal state and feedback scheduling after queued inputs are exhausted. The specific resource gap is a useful resident workload large enough to exercise memory traffic, with an effective CPU implementation and complete host transport included. Another association demo, a larger neuron drawing or an isolated GPU slowdown would not resolve either gap. No performance-victory X post is warranted from this campaign.
