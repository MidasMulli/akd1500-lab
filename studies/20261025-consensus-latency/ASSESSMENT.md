# Matched agreed-update latency on the Nano

**The CPU wins this small workload.** The AKD remains functionally exact, but an
actual memory-revision transaction takes **327.5 µs** versus
**21.1 µs** for the packed CPU neural replica: **15.5x** the latency.
The direct CPU agreement policy is cheaper still. This does not measure maximum
chip throughput, power or an architectural limit.

## Results (medians; p95 retained in SUMMARY.json)

| Path | Actual revision transaction | Source query + update + 16 recall checks |
|---|---:|---:|
| Native AKD, including transport/runtime | 327.467 µs | 5.304 ms |
| CPU packed neural replica, sparse fixed routes | 21.089 µs | 0.364 ms |
| CPU direct agreement policy, per-event clocked | 0.064 µs (timer-limited) | 0.928 µs (timer-limited) |

The clock-call calibration itself has a 64 ns median. The policy's raw 64 ns update
measurement is therefore not a resolved single-call cost. A separate resident
agreement+store batch measures **2.33 ns/update amortized**
(nine batches, including proposal loads, loop, function call and checksum). A mixed
source/query/transfer stream averages 2.59 ns/event.
These batches are explicitly different from executing a neural simulation and do
not imply nanosecond learning of arbitrary associations.
The batch is single-threaded with a tiny cache-resident table; completion means
the new answer is available to that CPU learner, not that stores have reached
external DRAM or survived power loss. No persistence/coherence claim is measured.

All three paths revise the two changed cues after their **first accepted teaching
attempt** in every tested cycle. Every revised cue passes 16 consecutive read-only,
teacher-answer-masked recalls, with no later same-cycle error. Across nine primary
runs per path (three seeds x three repetitions x six cycles): 108 actual revised-cue
cases, 648 accepted transfer calls, 2592/2592 protected final probes and 2592/2592
updated final probes. Most accepted transfer calls revisit already-correct/stable
memory; headline revision timings use the 108 actual changed-memory transactions.

Confirmation active time includes its 16 queries, so it is not the intrinsic time
to change memory. Native confirmation probes account for about 4.68 ms. From each
learner's first source query, raw observed completion including lightweight audit
bookkeeping is 5.313 ms native,
0.371 ms CPU replica, and
5.280 µs direct CPU policy.
Global readiness for both cues precedes the sealed interleaved order; the second
cue waits for the first cue's confirmation probes. SUMMARY.json and CASES.csv keep
that global elapsed endpoint separate from the per-learner endpoint shown above.

## Where native time goes

Actual-revision component medians (individual medians need not sum exactly):

| Observable boundary | Median |
|---|---:|
| Learning-mode configuration writes | 38.27 µs |
| Raw phase/input writes | 0.96 µs |
| Descriptor submission | 11.01 µs |
| Queue-release write | 0.93 µs |
| Release completed → observed device completion | 99.94 µs |
| Result fetch, SDK and returned-output copy | 176.21 µs |

Completion observation includes PCIe polling; it is not a pure silicon timer.
Result retrieval is the largest measured component. These boundaries separate
observable transport/runtime work from resident completion but cannot divide
the latter into pure PCIe and pure arithmetic by subtraction. The seven NPUs
execute sequential stages in this mapped program.

## Verification and execution

Full three-learning-bank readback plus checks costs a median
**4.34 ms per accepted operation** in the full-audit
qualification. This is excluded from primary timings. Primary cells verify every
output byte and complete final learning RAM; full-audit qualification verifies
every intermediate bank boundary. Fixed route RAM is checked at setup/closeout.
Disk serialization, initial learning, program loading and reference generation
are outside reported operational latency. No host writes occur during execution.

Nano ARM CPU2, 1.728 GHz min/max, Ubuntu g++13.3, O3/mcpu=native, serial rotated
arm order. No unrelated process or clock configuration was changed. 12 successful
physical cells, 29,448 forwards including qualification. 35 passing CPU/native
streams. Two failed/diagnostic attempts are preserved: an initial SIGSEGV and
bounded GDB replay exposed the new harness destroying its program buffer before
enqueue. The buffer lifetime was repaired; full-state qualification then passed
twice before timing expansion. This is a harness failure, not a silicon result.
All owned jobs are finished; card released. Study closed.

Archive: `7edb6a55787493596af0ff905f061440be166117b106edcc2aef47bca3761884`. SDK/card-free replay:

```sh
python -m unittest discover -s tests -p test_consensus_latency.py -v
python -m admission.latency_archive audit
```

## Interpretation

This provides no latency advantage for AKD on the four-cue task. Even the observed
resident-completion interval is slower than this CPU neural replica; host retrieval
adds substantial further cost. Functional native memory control remains demonstrated.
Any future hardware-efficiency claim needs a separately frozen workload that uses
more of the fabric and amortizes transport, with matched quality and direct energy
measurement. Do not infer an optimized full-chip ceiling from this small circuit.
