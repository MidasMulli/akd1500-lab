# Frozen world model with native AKD outcome memory

The new magnitude-preserving input passed the planned native series. In the four-context development retention test, the AKD recalled the correct outcome on **64/64 return queries across both physics settings**. In a separate error-only test, it corrected **16/16 returned changed-physics forecasts**, versus **0/16** for the A-only frozen control. The fresh held-out two-context hardware test also corrected **16/16 returned B forecasts**.

This is a bounded integration milestone: compact states from a frozen world model now feed a physically verified resident AKD learning-memory circuit. It is not a live GPU service, a closed-loop planner result, a demonstration of CPU impossibility, or a whole-chip capacity limit.

## All four cohorts, including the failures

| Cohort | Disposition | Physical execution |
|---|---|---|
| v2, joint10 | DEV passed, but HELD outcomes did not change across the two physics settings; the baseline was already correct. Usefulness was untestable. | None |
| v3, joint10 | Fresh changed-outcome HELD K2/K4 failed because distinct states shared the same sign-only key. | None |
| v4, joint14 | K2 passed; HELD K4 still failed from one sign-only alias. Full-float CPU passed. | None |
| Sparse magnitude, oct9 | Fresh DEV and jointly released HELD K2/K4 passed software gates; the planned native series below passed. K8 remains a capacity failure. | 7 sessions / 19,044 forwards |

The first three cohorts remain closed and preserved in `/Users/midas/astra_inbox/2026-10-28_akd_worldmodel_review/`. They are not counted as new independent confirmations. This study used one fresh cohort, two prospectively frozen encoder profiles and **zero new encoder fits**. Both profiles passed DEV; the frozen selection rule chose oct9 before HELD inspection.

## What changed

Sign-only encoding discarded small but meaningful differences in the world-model state. The new input represents each of nine prior PCA components with one of eight fixed magnitude bins. It preserves those differences while retaining exactly 32 connections per content neuron. PCA axes and scales come from the already frozen prior; neither was fitted to this fresh cohort.

The wider payload originally failed to map. Engineering qualification repaired the routing before fresh task data existed: a 1,024-byte window into the previous native output carries variable fields back to the controller; native fixed routing regenerates constants. Exact ternary/binary routing and compression of redundant comparison carriers let both profiles fit entirely on FNP3. The first fixed relay uses two-bit ternary weights; subsequent relays use binary weights. The normal/retire learning packets retain the compiler's application routes.

Failed mappings, import/venv mistakes, an unapplied patch attempt and the repaired source versions are preserved. Source qualification reached 16 under a prospective 12-to-16 engineering amendment; the original authority remains unchanged alongside that amendment.

## What runs where

- The producer runs the unchanged LONG202 world model on Ascent CPU and exports its state before and after a five-tick observation, its forecasts and subsequent simulator outcomes. No GPU/WM weights were changed here.
- The host applies the frozen PCA/bin encoder and supplies observations, later outcome teachers and fixed role/configuration clocks.
- Two learning banks and four fixed routing/readout banks occupy **six resident FNP3 NPUs**: controller `[1,3,0]`; routing `[2,1,0]`, `[2,2,0]`, `[2,3,0]`; eight-record content memory `[3,2,0]`; return `[3,3,0]`.
- Three fixed candidate probes query the outcome memory before the episode's feedback. The native controller's learned comparison gates revision/write traffic. Content recall returns through a card-side scratch window; the host does not decode and resend intermediate feedback.
- A prespecified host rule combines an exact memory answer with the calibrated model forecast: 75% memory one-hot / 25% model. Displayed forecast probabilities are not neuron confidence.

The four-context retention arm deliberately admits all outcomes to test storage separately from admission. The error-only arm admits discrepancies. In the latter, correct A forecasts have no native memory entry: correct WM fallback must not be called native A retention.

Resident means retained together on the device. The roles are host-paced; this does not establish concurrent NPU execution or operation with the host silent.

## Native stages and results

Every cell was CPU-qualified first, then launched once, independently audited and reviewed before the next cell. No physical failure or retry occurred.

| Cell | Test | Forwards | Result |
|---|---|---:|---|
| 00 | DEV K2, 12-episode routing/learning prefix | 684 | Full output/RAM/input/pointer audit passed |
| 01 | DEV K2, all outcomes admitted | 2,628 | Native A+B return recall 32/32 |
| 02 | DEV K2, error-only learning | 2,628 | Returned B 16/16; A remains correct through WM fallback |
| 03 | DEV K2, error-only A-frozen | 2,628 | Returned B 0/16; intended adaptation control |
| 04 | DEV K4, all outcomes admitted | 5,220 | Native A+B return recall 64/64, eight distinct stored keys |
| 05 | Fresh HELD K2, error-only learning | 2,628 | Returned B 16/16; no A harm |
| 06 | DEV K2, trained on A+B then frozen | 2,628 | Native A+B return recall 32/32; matches after both regimes are learned |

Total: **7 physical sessions, 19,044 reserved and observed forwards, 348 query episodes, 16,356 independently verified native feedback windows**. All full outputs, both complete learner RAM banks, defined actual inputs and descriptor addresses matched the sealed reference. All four fixed routing RAM banks remained unchanged. The card was released and no scientific job remains active.

Cold B encounters still fail before feedback: in the K4 arm, the first four B queries were wrong, then the remaining 12/16 queries in that first B phase were correct. Later B returns were 32/32. This is learning from supplied experience, not anticipation of an unobserved rule change.

The K4 native result is DEV, not native HELD K4. HELD K4 passed the independently audited software screen; the physical HELD cell is K2 only.

## Strong controls and remaining limits

Adaptive full-float CPU controls also pass. A native memory trained on both regimes and then frozen matches later returns. No speed/energy advantage is established: native wall times include full per-forward RAM capture and a CPU shadow used for verification. They are not substrate latency measurements.

The cohort deliberately admits contexts whose actual outcomes change with physics, using simulator eligibility rather than model error. K prefixes are nested; episodes repeat cached context/gain computations. Counts above are repeat queries, not that many independent scenes or natural video samples. The producer preserves every rejected candidate and a bit-equality cache guard. Report the selected challenge distribution rather than a deployment success rate.

The K8 diagnostic has 16 distinct keys and only eight slots. It fails the memory-capacity gate without cue aliases. Eight is this configured memory's capacity, not the physical maximum of the AKD1500. No expansion or new cohort was launched to hide that result.

## Measured visual

`visual/index.html` is a read-only 3D replay. It uses the prospectively fixed first admitted DEV scene, 202610280100022 / WAIT / P4, across A and B. RGB inputs, imagined frames and final simulator frames come from the producer's fixed A/B media export. The chip stores sparse associations and outcome tags, not these pictures.

The eight displayed output bits and full before/after connection records come from the native K4 trace. Input activity and overlaps are reconstructed from the actual captured input bytes and pre-update RAM; all 162 selected content events are independently checked. No individual controller-unit spikes or electrical timing are invented. Shape, placement and replay timing are schematic. Selection jumps over intervening contexts; full raw traces remain in the archives.

The interactive replay and film show the all-admit retention arm. The separate error-only/frozen comparison is reported numerically above rather than falsely attributing that admission rule to the displayed arm.

## Next assessed target — unlaunched

Run a resident corrector on fresh causal states from the Nano's GPU world model, then test a native budget/replacement policy when new regimes exceed available slots. Keep query-before-feedback timing, unseen contexts, useful retained predictions and closed-loop actions separate. Qualify source geometry and strong controls before hardware. This report authorizes no further cohort, encoder, physical session or deployment.

Primary receipts: `FINAL_ASSESSMENT.json`, `NATIVE_LEDGER.json`, `AUDIT_DEV.json`, `AUDIT_HELD.json`, per-cell native audits/task results, `ENCODER_FREEZE.json`, `NATIVE_PROTOCOL.json`, and `visual/DATA_QA.json`. Every native raw byte is retained in a readback-hash-verified `NATIVE_XX.zip`. The local review package is not a GitHub publication.
