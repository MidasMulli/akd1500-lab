# Useful recall is not the same as repetition

**Software replay only; no new hardware run.** The inherited circuit still
reproduces its published 480/480 held-out result. Repeated distractors expose
limits of both its admission heuristic and its lack of memory reclamation.
These results forecast the inherited circuit on new inputs; they do not add
measured AKD behavior to the preceding physical study.

## Results

Five initialization seeds, four circuit arms, three sealed schedules: 60 software
traces and 28,880 replayed forwards. Initialization 114101 passed its invariants,
inherited-result checks and saved-array audit before the other four were run.

| Schedule | Intact circuit replay | Old-key recall | Later-key recall | CPU repeat LRU | CPU ordinary LRU | CPU unbounded |
|---|---:|---:|---:|---:|---:|---:|
| Published original |480/480|150/150|330/330|480/480|330/480|480/480|
| Repeated distractions, spaced later inputs |150/480|150/150|0/330|150/480|330/480|480/480|
| Repeated distractions, repeated later inputs |170/480|150/150|20/330|330/480|330/480|480/480|

One initialization learns a later association in the repeated-later-input case;
its 20 correct queries are included. Do not replace that case with an all-seed
failure claim. The always-open circuit has 170/480 in each schedule; frozen
consequence banks and frozen manager have 150/480. These are repeated checks on
one fixed workload per schedule across initialization seeds, not independent
environments or 480 novel classes. The six queried keys are already observed.

The CPU controls receive the identical raw observation/outcome stream and have
an eight-entry limit where stated, not a byte or electrical-cost match. They do
not receive future-use tags, feature vectors or native record IDs. Ordinary LRU
loses old keys but retains later keys here; a perfect general memory-management
solution is not established by that tradeoff.

## What the counterexamples show

1. Recency/repetition is not a measure of future usefulness. The current manager
   admits repeating distractors and refuses spaced observations. That behavior
   follows the engineered rule, even when its recent-key state is learned.
2. Opening the gate is insufficient once earlier repeated inputs have occupied
   the finite bank. Repeating later inputs changes the gate, but it does not
   introduce a qualified eviction/replacement mechanism.
3. Old-key recall survives while later associations are mostly missing. The
   observed software loss is not evidence that old memories were corrupted.

Here "useful" means queried again in this sealed suffix. Future query selection
is an evaluation choice, not a relevance label discovered by the circuit. These
counterexamples cannot establish that a new policy could predict arbitrary
future relevance. A useful next policy needs grounded feedback, such as later
prediction errors, and a specified objective and information budget.

## Verification and scope

Both saved numeric archives replay exactly: every input, output, packed potential,
fixed feature vector and learning-bank RAM boundary. Fixed forwarding records,
32-connection budgets, disabled consequence-bank queries and zero-input blocked
teachers pass their invariants. The original physical numeric archive separately
still passes (31 qualified sessions/8,828 measured forwards); that count is not
increased by this screen. There were no failed software cells in this series.

The only inherited-code change is an optional explicit event list to the replay
function. Default scheduling and the physical launch path are unchanged. No SDK,
device, compiler or physical discovery was used by this screen. Its reports bind
the source and protocol hashes, complete plans and numeric archive hash.

```sh
python -m admission.relevance audit results/20261025-relevance-gate.json
python -m admission.relevance audit results/20261025-relevance-expansion.json
```

To reproduce without overwriting saved attempts, choose new output paths:

```sh
python -m admission.relevance run results/reproduced-relevance-gate.json --seeds 114101
python -m admission.relevance audit results/reproduced-relevance-gate.json
# Inspect that gate before running the remaining initializations.
python -m admission.relevance run results/reproduced-relevance-expansion.json --seeds 114201 114301 114401 114501
python -m admission.relevance audit results/reproduced-relevance-expansion.json
```

## Next focus

Build a resident experience memory for future prediction: native comparison of a
saved prediction with a later actual outcome, then a qualified native path from
that error into admission and bounded memory reuse. The first comparator gate
should enumerate correct/wrong/missing/ambiguous predictions, both actual outcomes
and query/teacher phases. Prediction must precede the outcome; the host must not
compute an importance tag, admission decision or record repair.

A fixed prediction-error criterion remains engineered. Qualifying that criterion
does not mean a management objective or utility policy was learned. Subsequent
claims require learned state to change future admission, verified capacity reuse,
and lower future prediction error against frozen, shuffled-feedback and strong
CPU controls. Hardware clocks, native credit routing and replacement all remain
separate admission gates. No later physical study is authorized by this report.
