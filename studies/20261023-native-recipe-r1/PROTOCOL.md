# Prospective conservative-gating repair

The original native-recipe screen remains closed at its failed CPU gate. It
reduced feedback forwards but was not reliably noninferior on both old and new
probes. This is a separately named development repair, not an overwrite/retry.

Keep the original corpus, complete training/probe schedules,128-neuron budget,
32-connection native rule and all quality/cost gates. Restrict this repair to
the unsplit four-class layout so a population-layout change cannot explain it.
Compare two predeclared conservative policies on development seed98401 only:
teach on wrong label OR margin<4 OR highest match<31, and the same with match<32.
Old acquisition remains always taught. The higher thresholds address the observed
loss of adaptation from stopping teaching too early. No new synapse math is used.

Select only policies passing the original gate, by lowest adaptation teaching
count, then policy name. If neither passes, stop this repair. Freeze the selected
policy before forecasting fresh initialization seeds98404/98405/98406. Require
all three confirmation forecasts to pass before native confirmation admission.
No further parameter search or lowering of accuracy thresholds in this study.

All prediction requests precede feedback. Host observes native query scores and
chooses whether to submit a later labeled native teaching job. CPU never computes
or writes connection updates. Native smoke and full development pair must pass
exact output/full-RAM checks before confirmation seeds. Measure gating CPU time,
mode-transition count, query/teaching cost and complete feedback-event latency.
Probe latency and acquisition overhead are reported separately from adaptation.

Maximum10 physical sessions,35000 forwards,180 seconds per process, with source,
fixture,binary freezes and explicit admissions. No RAM-only restore. Preserve any
first failure, inspect before repairs, and enforce the saved common hard stop.
