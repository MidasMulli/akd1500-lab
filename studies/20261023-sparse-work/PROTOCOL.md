# Native sparse-work and input-scan discriminator

Direct operator authorization: "Go" after the next-stage hardware plan. New bounded
study; previous execution and ablation studies remain closed. Synthetic mechanism
characterization, not an application or an electrical-efficiency claim.

Freeze seed 98201; 64/256 logical neuron records; 128/512/1024 binary input bits;
8/32/64 selected connections per record. Fixed half-active inputs and the same
deterministic input stream for all connection counts and widths at each dimension.
Connection sets are nested as the connection budget grows. Fix the learning bank
at FNP3 (1,3,0); reject FNP2/external-memory or multi-pass substitution. Compact
output adds a fixed FNP3 (3,3,0) two-count readout using the previously validated
intermediate-learner adapter. Keep its memory and the learner state observable.

## Competing explanations

* Full-input scan: initial-inference service cost scales with input words times
  logical records, even when selected connection count is fixed.
* Selected-connection work: changing connection count at fixed dimension/width
  changes service cost; enlarging an otherwise sparse input need not.
* Transport/readout: packet or result volume dominates. Compare raw scores and
  fixed two-count outputs; the extra readout is an explicit graph difference,
  so do not infer an absolute raw-versus-compact compute speedup.

## Staged admission and checks

Compiler/virtual preflight covers all 36 path/cell combinations, with exact
matrix/reference outputs, constrained mappings and decoder checks. No physical
device discovery in preparation. Preserve preparation failures.

Start at width64 with (inputs,connections)=(128,8),(1024,8),(128,64), both output
contracts. Smoke: 16 inference events, eight teaching events, eight retained
queries, followed by the identical eight-query suffix three times serialized
and three times as preloaded bursts. Inspect each run's full outputs, every
learning-RAM snapshot, initial/final fixed-readout memory, idle/configuration
counter controls and independent integer recount before admitting the next.

After all six smokes pass, admit their full streams: 96 initial queries,
32 teaching events and 16 retained queries plus the same 48 suffix events.
Inspect those before the remaining dimensions/widths/connection budgets.
Admit two further temporal repetitions of the six anchor cells only after the
screen passes. Maximum72 sessions/16000 forwards/180 seconds per owned process;
the saved authority has a scientific cutoff and a hard stop two minutes later.

Numerical mismatch, RAM mismatch, unexplained mapping/substitution, advancing
idle counter or another device owner halts expansion. Preserve the first failure
and diagnose before a prospective repair/admission. No silent retry. Do not
restore only visible RAM to resume learning: every session starts fresh and
replays its exact prefix. Retain source/fixture/binary freezes and owner receipts.

Inference and learning counters have different buffer-end scopes: report them
separately. DMA monitor cycles are not arithmetic-only or per-neuron traces.
One-chip temporal repetition is not a multi-chip estimate. Dense host transport
remains in use. No firmware, clock, installed-driver or undocumented-address sweep.
Any two-bank or CPU application stage requires its own checked prospective cells.
