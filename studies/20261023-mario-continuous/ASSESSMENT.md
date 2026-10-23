# Continuous interface verified; first matched pair regressed

The [protocol](PROTOCOL.md) prospectively replaced 90-frame commitments with
24-frame motor primitives and corresponding short outcome labels. One native
bank now repeatedly queries both candidates, acts, learns from its actual
consequence and queries again. Neural state remains resident across the four
environment episodes. No emulator reset occurs inside an episode.

| Development level | Learned advance | Frozen advance | Learned decisions | Frozen decisions |
|---|---:|---:|---:|---:|
| 1-1 | 538 px | 538 px | 16 | 16 |
| 2-1 | 173 px | 173 px | 6 | 6 |
| 3-1 | 428 px | 445 px | 13 | 36 |
| 4-1 | 279 px | 398 px | 11 | 11 |
| Total | **1,418 px** | **1,554 px** | **46** | **69** |

All eight episodes ended in death. The predeclared first-pair gate stops
expansion when learned total displacement is below frozen, or fewer learned
episodes survive. **Status: STOP_DEVELOPMENT_REGRESSION_CONFIRMATION_UNOPENED.**
Seeds 84102/84103 were compiled but not run on hardware; the separate confirmation
starts were not opened. The earlier failed sealed study remains unchanged.

Before physical execution, the stateful Python reference matched every score
and RAM snapshot of a previously published 1,376-forward native stream. The
virtual mapping checks passed, then a 32-input mixed smoke passed. The full
physical arm was independently recounted and freshly replayed before its
matched frozen control. Both native trajectories exactly matched the CPU
reference forecasts, including every captured RGB frame.

The three physical sessions contain **3,385 native forwards**. Full native,
C++ and independent Python outputs/RAM agree throughout; all protected queries
leave visible RAM fixed. Frozen retains its entire initial RAM. The learned
arm makes **507 online connection replacements** across its 46 decisions,
in addition to offline preparation. Four game episodes share one resident
neural state per arm; this is not a hidden-state restore or power-cycle test.
The device was released and admissions closed after the failed gate.

Training contains 96 valid contexts, both action outcomes, 180 distinct encoded
inputs and no contradictory encoded labels. Only **4/192 outcomes were unsafe**,
all for walking after the same supplied prefix family in levels 1-1 and 3-1.
Class balancing repeats those outcomes; it does not broaden hazard coverage.
This limits interpretation. It is not a controlled causal proof that coverage
alone explains the regression. The authored RGB/time representation, short
consequence objective and motor primitives also remain candidate limitations.

CPU-only constant controls on these same starts were:

| CPU policy | Total advance | Alive at cap | Completed levels |
|---|---:|---:|---:|
| Constant A0 (short jump/right) | 1,430 px | 0/4 | 0/4 |
| Constant A1 (right) | 826 px | 1/4 | 0/4 |

They are model-free emulator baselines, not physical AKD control arms or CPU
efficiency measurements. The survived constant episode did not complete its
level. One seed on one chip and four development starts cannot qualify a
reliable task advantage. A continuous interface and exact native learning are
mechanism results, separate from the negative gameplay outcome.

The [paired 4K60 capture](../../visual/mario-continuous-4k60.mp4) displays actual
native match responses, decision margins, chosen actions, feedback and selected
RAM connections. Separate recordings are aligned at episode starts. Each view
preserves its own timing and is capped at its own episode end; a darkened view
means that arm died. Later-episode responses are never overlaid on earlier
ended gameplay. Gold recent-edit outlines last 0.3 seconds and are labeled
with age. Electrical spike timing and physical neuron layout are not measured.

The data, compiler producer and protocol were source-frozen before collecting
the new training outcomes. The controller/transport source was authored to that
protocol after collection and frozen before physical execution. No scientific
parameters were tuned after outcome inspection. All source/fixture hashes were
rechecked on the device host. Evaluation and visualization code were built
after recording and do not influence the acting path. Failed visual render
attempts and a clock-cap correction remain preserved locally; raw scientific
records were unchanged.

The next stage should develop and verify richer hazard/outcome coverage,
local motion features and action-duration adequacy before another prospective
qualification bank. Repeating this pipeline over more seeds is not justified
by its failed first-pair gate.

[Results and raw archive](../../results/2026-10-23-mario-continuous.json),
[reproduction and evidence boundaries](../../mario/continuous/README.md).
