# Native synaptic state controls what another bank learns

**Qualified: a native plastic manager controls memory admission, and this preserves
space to learn later useful associations while retaining old records.** Across
five initializations the protected bank answers120/120 useful-memory revisits and
480/480 queries in a fresh ordering. Strong CPU controls also succeed. This is a
bounded native-coordination milestone, not an efficiency or autonomous-agent result.

## Measured comparison

| Endpoint, five initializations | Native manager | Always-open policy | Trained-frozen consequence banks |
|---|---:|---:|---:|
| Final revisit: two old/four new keys |120/120|44/120|40/120|
| Held-out ordering |480/480|170/480|150/480|
| New useful inputs, including cold queries |260/320|14/320|0/320|
| Every query, including cold/distractor/transient |970/1300|520/1300|300/1300|

Every held-out run uses the same sealed96-key ordering. These are repeated checks
across five seeded initial weight sets, not480 independent environments. The
closed twelve-key synthetic stream is deliberately suited to a repeat-admission
policy; utility is an experimental workload choice, not a semantic importance
judgment learned by the chip. The bank forecasts a binary consequence associated
with a raw key, not a future video frame, an action, or a complete world state.

Original record IDs/weights stay unchanged10/10. The open control also retains its
old records. The measured advantage here is preserving **unused capacity** for later
repeated useful keys; do not describe it as preventing observed corruption of old
memories. One open-control initialization learns one later key; that success is
included in170/480 and14/320, rather than hidden behind the first-seed result.

## What is actually native

Seven sequential FNP3s in one resident program:

| Role | Physical placement | Plastic versus fixed records |
|---|---|---|
| Raw-observation encoding |[1,3,0]|Fixed|
| Synaptic recency manager |[2,1,0]|1 plastic,31 fixed helpers|
| Permission/feature gating |[2,2,0]|Fixed|
| Always-update consequence bank |[2,3,0]|8 plastic,40 fixed helpers|
| Native bridge |[3,1,0]|Fixed|
| Protected consequence bank |[3,2,0]|8 plastic,24 fixed helpers|
| Prediction/permission/raw-key relay |[3,3,0]|Fixed|

The manager record has25 common connections and7 variable key connections. The
fixed common input keeps it eligible for the existing native learning rule under
the frozen settings. On an observation it returns a match against its old weights,
then native swaps store the new key. Every manager-learning boundary holds the
exact expected current-key connections. Its recent-observation state is in
native learning RAM, not in an acting CPU variable or the previous-key input field.
That previous-key field is disconnected from the native encoder.

The query returns both a consequence prediction and the manager's pre-update match.
The teacher input points directly to this native output, including its permission
bit and raw key. Nano writes the uniform phase word and supplies the actual outcome
after the prediction. Native fixed routing decides whether the protected learner
sees a nonzero event vector. Zero input leaves its captured RAM unchanged. No host
code tests the permission to choose a bank, record, learning mode or replacement.

The host still schedules every forward and supplies predetermined per-bank phase
clocks: manager learns during observation queries, consequence banks learn during
teachers. Those clocks do not depend on permission. During held-out queries,
consequence RAM stays unchanged but the manager continues tracking observations.
This is not whole-chip read-only inference, indefinite self-scheduling or operation
with no external observations. Fixed forwarding helpers remain unchanged throughout.

## Causal controls and full verification

First-seed short gates qualify native8 refused/8 admitted teachers, open0/16 and
closed16/0. Bootstrap recall passes before pressure. Native pressure passes before
the open/frozen controls. First-seed held-out and causal controls pass before the
four further initializations. Every raw saved input/output packet, source pointer,
descriptor, all three RAMs and final fixed-layer RAM matches independent Python
bit-set and C++ packed replay. This establishes event-boundary behavior; it does
not expose electrical spike times or hidden internal registers directly.

| First-seed held-out control | Correct | Interpretation |
|---|---:|---|
| Intact native manager |96/96|All six useful associations available|
| Manager trained then frozen |30/96|Old associations remain; new admission fails|
| Returned permission severed after bootstrap |30/96|Same manager learning RAM as intact arm; downstream benefit removed|

All24 qualified sessions/7,780 forwards pass, plus the separate prerequisite's
seven sessions/1,048 forwards. Nine validly rehashed semantic corruptions are
rejected, covering raw input, manager synapses, permission in/out, phase clocks,
future-label leakage, teacher source, protected recall RAM and fixed policy RAM.
The actor programs once per run; no weight restore or host semantic bank choice
occurs. The reported host_bank_decisions counter excludes the predetermined per-
bank phase-clock writes; it counts data-dependent choices from the manager response.
Instrument mode/call counters are source journals, not an electrical bus trace.

## Strong CPU results and limits

| CPU control on the intact raw stream | Held-out ordering | Every query |
|---|---:|---:|
| Unbounded key/outcome dictionary |480/480|1240/1300|
| Capacity-eight repeat-admission LRU |480/480|1000/1300|
| Capacity-eight ordinary LRU |330/480|1050/1300|

These receive raw keys and outcomes; no native record IDs or encoded features.
Capacity is matched by entry count, not bytes or electrical cost. CPU dictionaries
learn cold associations faster, and ordinary LRU outperforms this native policy
over the entire mixed stream despite poorer final useful-memory recall. The CPU
can reproduce the management rule. No speed, power, energy or CPU-contention
advantage has been measured by this correctness instrument.

The repeat rule and fixed code are designed. Only its recent-key state and the
consequence associations are learned. Repeating irrelevant inputs would also be
admitted; relevance, eviction, consolidation, credit assignment, general utility
optimization, power-loss persistence and arbitrary outcome reversal remain open.
One plastic manager record is a small mechanism, not an emergent strategy or proof
of whole-chip limits. All seven NPUs run a sequential graph; this is not a measured
concurrency claim or an exhaustive32-NPU stress test.

## What stretches beyond the documented baseline

The documented edge-learning workflow configures a final binary-input/weight
classifier. We reuse that existing learning rule, competition and class-label
primitive. The contribution tested here is an own compiler/runtime adapter that
keeps **three resident plastic banks**, including intermediate banks, and connects
native synaptic state to downstream write admission and subsequent predictions.
It is beyond the stock final-classifier workflow in this project; we have not
established a global first, an undocumented new instruction, or that BrainChip
has never built a similar circuit.

[Official learning constraints](https://doc.brainchipinc.com/user_guide/akida.html#learning-constraints) ·
[official unsupervised example](https://doc.brainchipinc.com/examples/edge/plot_2_edge_learning_parameters.html)

## Preserved failures and publication boundary

The prerequisite preserves a rejected CPU gating design, failed placement and
auxiliary assertions, and one zero-forward physical aligned-write failure. The
final series has no failed physical model trace. Two outer orchestration stops
(absent local log directory; indexing a missing run() return after a passed cell)
are recorded in Amendment01. Completed cells were kept and never rerun to hide
those failures. Scientific sources, numeric evidence and public tooling are
packaged; licensed runtime/program images stay private.

The32-second4K60/1080p60 film replays selected hardware events. Stored connection
changes and returned response bits are measured; intermediate features are derived.
Coordinates, camera and flow animation are schematic. Teacher-time paths can gate
returned response bits, so a missing displayed bit is not an electrical inactivity
measurement. The follow-up post can fairly show native learning controlling native
memory formation, with the designed rule and CPU parity stated. Nothing was posted
or saved into X by this study.

```sh
python -m admission.archive audit
python -m admission.mutations
```
