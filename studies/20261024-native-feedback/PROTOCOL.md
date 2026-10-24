# Bounded output-to-input feedback, prospective protocol

Authorized by the user's request to build the bounded native feedback loop,
check every stage, qualify what executes while the Nano worker is quiet, and
capture measured activity and connection changes. This is a new study; earlier
closed studies are not reopened.

## Question and boundary

Can each queued AKD forward consume its predecessor's dense output directly
from chip scratch SRAM, without a host reading/re-encoding/copying that output?
Can native FNP3 learning continue over this dependent stream and alter later
query responses? What stops after the finite queue is exhausted?

This is a finite descriptor schedule, not indefinitely recurrent routing or
self-scheduling. Only the experiment's host worker sleeps; the Nano OS remains
active. Quiet intervals have no driver reads and no writes after one queue
release. Device-local scratch and FNP3 SRAM are distinguished. No energy,
CPU-superiority, useful-task, unseen-world or global novelty claim is admitted.

## Frozen model and controls

128 binary inputs and 128 learning records, each selecting 32 connections.
Four disjoint 32-bit state groups. Each output group initially connects 16
features from its predecessor and 16 from other groups. Output threshold 13,
learning threshold 15, one constant class 0. Initial/minimum plasticity .5/.25,
decay .125, competition .25. A fixed second FNP3 passes the 128 binary outputs.
The ring transition is programmed. Native learning may sharpen its records;
this is not learning a new transition graph from outcomes.

Seeds 109101, 109201, 109301. Candidate horizons 1,2,3,4,5,8,16,32. Each run
starts from the compiler-frozen initial state. Every output and full terminal
learning SRAM must match separate Python and C++ exact-rule replay. Fixed
readout SRAM must remain unchanged. Prefix-run state snapshots are independent
replicas, never mislabeled as simultaneous probes of a single quiet run.

Arms: native closed feedback; learning disabled from initialization; repeated
seed instead of feedback; and severed feedback (zero input after first step).
Four post-loop partial-cue queries test whether changed connections alter
responses. Query construction and their interpretation are frozen before device
admission. No autonomous discovery claim follows from rehearsal of a seed.

## Stages and gates

1. Compiler/virtual checks: dense binary I/O equality, two FNP3 placements,
   packet routing and numeric reference. Failures retained.
2. Frozen fixed network at 1,2,4 steps, inspected one at a time. At two steps,
   a different output must establish the dependency, rather than the repeated
   seed's constant response.
3. Native learning at 1,2,4 steps, each independently checked before expansion.
4. Remaining prefix horizons and 32-step repeated-seed/severed controls.
5. Only after primary qualification, replicate the 32-step four-arm comparison
   in the two remaining initializations.
6. Portable audit and recorded visual. Negative/partial results still delivered.

Per physical run, prepare the single seed, zero-control payload, all descriptors
and initial model before release. Dense output header is 32 bytes; feedback
input pointers target preceding output payloads. Verify descriptor changes are
only the input address. No future state payload is uploaded. Disable runtime
pipeline speculation. Sleep 5 ms after release, require complete at first wake,
then fetch/check. Record a later 20-ms quiet interval after completion and prove
output/job state and learning SRAM stable. A failed first-wake bound is retained
as failure, not silently extended. Wall times are observation bounds, not
per-step device timestamps.

## Operational bounds

At most 40 physical sessions, 4096 native forwards including post-loop queries,
and 200 CPU repeats per cell. One owner at a time using nonblocking
/tmp/akd1500-research.lock plus fuser. Every run gets an immutable owner receipt,
source hashes and raw files. No firmware, clocks, driver installation or
undocumented cyclic-router writes. Each owned process has a 30-second external
timeout; only owned processes may be stopped. Scientific cutoff one hour after
authority creation, hard stop two minutes later. Do not expand after a mismatch
without preserving and diagnosing the failure and making a prospective repair.
