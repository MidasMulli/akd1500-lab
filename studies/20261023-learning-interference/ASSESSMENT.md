# Two native learning banks preserved their solo states under combined teaching

Measured: 16 sessions, 10,016 forwards and 20,032 complete per-bank/per-event RAM
captures. Four smoke arms replayed the full512-event prefix before a short probe;
12 full arms covered masks 00/01/10/11 across initialization seeds 97101/97102/97103.
Every native output and bank state matched the independently implemented rule.

The resident graph has five FNP3 assignments: recognition(1,3,0), fixed pairwise
comparison(2,1,0), fixed categorical message(2,2,0), routing learner(2,3,0), fixed
four-count readout(3,1,0). No host intermediate activation was injected.

## Matched controls

Each arm started fresh and replayed the same512 native training events; no
RAM-only restore was used. The128-event test used the same noisy inputs and
shared identity labels. Before physical admission, the CPU gate required the
categorical messages reaching the second bank to match across masks. Native
outputs and complete RAM then matched that same model. Intermediate messages
are reconstructed from the verified rules/state, not directly probed wires.

| Seed | A connection-edit jobs, A-only / both | B connection-edit jobs, B-only / both | Jobs editing both banks |
|---|---:|---:|---:|
|97101|80 /80|8 /8|4|
|97102|84 /84|8 /8|7|
|97103|86 /86|8 /8|7|

These edits include actual connection-bit changes, not only threshold changes.
At most one record changed per bank/event. A's entire captured RAM trajectory
in 01 equaled A's trajectory in 11. B's trajectory in 10 equaled B's in 11. Query-only
banks changed no state. All three fixed communication/readout RAM regions were
identical before and after every session. This is stronger than matching final
task scores alone, but is specific to this graph, common teacher and histories.

## Timing and its boundaries

All arms kept the same DMA buffer-end control925696. Idle reads stayed fixed;
configuration counters stayed fixed during each event. For seed 97101, test
median service cycles were22256(neither),22354.5(A),22275(B),22376(both). These
small approximately additive increments do not demonstrate overlapping updates.
The graph is feed-forward and its second learner depends on the first message.
Same-job edits are not proof of physical clock-for-clock simultaneity.

Artifact transfer/source snapshot preparation overlapped late confirmation runs.
Host latency is descriptive, not a clean comparative system benchmark. The first
numerical export contained12 complete arms and passed recount before the remaining
confirmations finished; that provisional archive is retained privately. The final
archive contains all 16 planned runs and the strengthened export gate requires
closed admissions and the complete treatment set. No native failure was retried.

## What this does not establish

The shared identity teacher is not independently addressed teachers. The second
bank's old destination mapping may change, so this is not an application-quality
improvement. Arbitrary recurrent circuits, independent input streams, all-chip
plastic capacity, physical update overlap and an electrical advantage remain
unresolved. The CPU contextual comparator reproduced all outputs and final RAM;
the both-learning CPU path took about5.07 µs/test event on this host. That is not
a chip-level speed comparison: the native adapter includes transport and fetch
sleeps, and electrical power/whole-host IRQ costs are not qualified.

Closed; no owned native jobs remain. Exact executed source/binary snapshots are
retained privately; post-closeout audit hardening is not a hardware rerun.
Recount: `python -m metal.interference audit-public` and
`python -m metal.cpu_compare audit-public`.
