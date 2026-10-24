# Delayed-feedback native learning recipe, equal resources

Operator authorization: "Go" on improving native execution/learning and auditing
the internal programmability boundary. Previous hardware studies stay closed.
This study changes the learning policy, not the internal synapse-update engine.
Every connection update is calculated by the AKD. Host selects teaching events
and labels; autonomous on-chip policy control is not claimed.

Freeze 512 binary inputs, 128 total neuron records, 32 selected connections per
record, learning parameters(.5 initial,.25 minimum,.125 decay,.25 competition).
Initialization seeds98401/98402/98403; corpus98410; fixed noise schedules and
checkpoint probes in `recipe/design.py`. No tuning after observing the forecast.

Four arms: stock continuous teaching, error/confidence-gated teaching, a stock
stable/adaptive population split, and the same split with gated adaptation.
The split uses two logical64-record populations in one FNP3 NPU, not two physical
NPUs. Unsplit uses four classes; split uses eight native teacher groups paired
into four semantic outputs. Initial per-class connection pools are permutations
of the same owned weights. All arms have identical total connection memory.
Both split arms use the same explicit training-regime transition to select the
adaptive teacher group. Inference uses all available semantic class matches and
does not receive a regime/domain label or probe identity.

The stream contains512 old-regime training examples and256 changed-regime
examples, with noisy held-out old/new probes after0/16/32/64/128/256 new examples.
New signatures are correlated variants of old signatures with shifted semantic
labels. Every prediction request precedes teacher feedback. Baseline uses a
separate native teaching forward after each returned prediction; gating may
skip that later forward. Future labels are not supplied with prediction jobs.

Gated update: wrong prediction OR semantic margin<2 OR highest match<29; old
acquisition is always taught. No raw synapse writes or RAM-only learning restore.
Gate compares each gated arm to its corresponding same-layout stock control:
old-prefix accuracy>=115/128; final old/new>=109/128; final accuracy loss<=2/128
on either probe; adaptation teaching jobs<=75% of stock. Report failed gates.
Sample efficiency is the first fixed checkpoint reaching109/128 new accuracy;
no interpolation or post-hoc checkpoint selection.

Stages: CPU/reference forecast first; compiler/virtual preflight next. Admit only
development-seed native smoke pairs after reading the forecast. Smoke replays
the complete old acquisition then the first16 adaptation events and fixed probe
subset. Check every native query output, feedback output, full RAM after each
logical event and protected records. Inspect before full development streams,
then before confirmation seeds. No automatic expansion. Maximum20 native
sessions,100000 forwards,180 seconds/process within the saved authority window.

Record host gating cost, mode changes, query/teaching device service separately,
end-to-end event time including transport, and all gates. Counter scopes differ
between inference and teaching; no cross-mode cycle speed ratio. Same quality
with fewer native jobs is a recipe result, not CPU or electrical superiority.
Preserve failed attempts, source/fixture/binary freezes and exclusive ownership.
