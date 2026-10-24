# Resident scaling stopped at the mixed-resource memory gate

Closed October 23, 2026 Central, before the saved scientific cutoff. Six owned
physical sessions completed **663 actual forwards** and saved **796 complete
per-bank RAM captures**. All jobs unprogrammed and released the device. The
requested/admitted sample accounting was 1,440; failed sessions did not execute
their remaining requests. No four- or six-bank physical session was admitted.

This is a failed new scaling path, not a regression of the previously verified
two-learning-bank, five-FNP3 graph. The proposed new graph moves communication
into fixed CNP resources, leaving FNP3 resources for additional plastic banks.
Six banks (384 learning records) and 17 total NPUs map in one compiler pass.
That mapping is **compiler evidence only**, not demonstrated native learning.

Follow-up clarification: the separate inference-only diagnostic
`20261023-fixed-bridge` also fails the compiler-produced FNP→CNP graph, without
learning or custom RAM edits. The learning adapter is therefore not the sole
possible source of the composition problem. Its failed traces remain unchanged.

## Prospective task and admission gates

Each bank has 64 records, 512 binary inputs, 32 selected connections, four
classes and threshold 27. The first memory learns noisy sensor signatures;
later memories learn different class permutations. A fixed native pairwise
comparison/encoder relays categorical messages between memories. The host
supplies the initial sensor event, scheduled teaching labels and learning-mask
changes; it does not inject intermediate neural activations.

One synthetic world and initialization seed 99101 passed the prospective CPU
repair/retention gate at one, two, four and six banks. Every CPU forecast gave
64/64 old, 16/64 changed-before, 64/64 repaired and 64/64 old-retention decisions.
Frozen and wrong-bank controls did not repair the changed signals in those
forecasts. These were not physical confirmation runs. Fresh initialization
seeds 99102/99103 were compiler-prepared but never admitted to hardware.

## Preserved execution outcomes

All attempts replayed the complete prefix from a fresh program, including
hidden selector history; there was no RAM-only resume. Each two-bank failure
halted expansion. Corrections were recorded prospectively, with distinct source
freezes and fixtures, and changed the adapter rather than the task or seed.

| Implementation | One-bank native check | First mixed-bank RAM divergence | Actual mixed-bank forwards |
|---|---|---|---:|
| 01: restored prefix auxiliary words | 176/176 exact output and RAM | Input 129, downstream bank |129|
| 02: retained intermediate compiler words 1/2 |176/176 exact|Input 1, upstream bank|1|
| 03: retained only compiler word 1 |176/176 exact|Input 5, upstream bank|5|

Attempt 01 has every bank snapshot through its first failure, but no saved final
output prefix: that instrument initially flushed output counts only on normal
completion. Later instruments flush the prefix on memory mismatch. This missing
measurement is preserved, not reconstructed and presented as a measured output.

At attempt 01's first downstream update, the reference changes record 6 while
the native RAM changes record 13. A private diagnostic finds the native update
consistent with a categorical-zero input rather than the upstream reference's
categorical-three message. This is an interface hypothesis, not a wire trace.
The cause of the original communication mismatch remains unresolved.

## Compiler field diagnostics and limits

The inference compiler and terminal-learning prefix disagree in auxiliary record
words. Copying the entire learning-prefix record into an intermediate layer was
not sufficient; copying inference-only auxiliaries was also not sufficient.

Controlled compiler-only differentials show:

* Record word 2 changes from 32 to 16 with `num_weights`; zeroing it is not a
  justified way to retain the learning configuration.
* The observed high half of word 1 is zero for one class and 16 for two/four/eight
  classes. An earlier hypothesis that it equaled `units/classes` was rejected
  and preserved. The experiment does not identify the complete bit-field meaning.
* Plasticity/competition changes also affect learning-track words. The software
  must preserve the full learning/output contract, not only weights and a few
  mode words. This is an unvalidated interface/adapter path, not a chip capacity ceiling.

The last bounded correction still failed. No further native retry or alternate
recipe was substituted in this campaign. No larger learning, useful-application,
CPU-efficiency or electrical-energy claim is qualified. Existing two-bank
FNP-only results remain separate. The new post/demo gate did not pass.

## Separate image partition branch

The bounded two-1,600-filter-bank image attempt failed while adding the SDK
`Concatenate` layer: its IP version did not match the AKD1500 model's layers.
Both raw and compact representations retain that failure. Neither reached a
physical session. The earlier 3,200-exemplar quality and CPU results are unchanged.

## Evidence and next gate

Public numeric traces, successful prefixes, failed prefixes, actual mappings,
compiler diagnostics and a hardware-free recount are in `resident/` and
`results/2026-10-23-resident-boundary.{json,npz}`. Executed sources, full freezes,
compiled programs and raw receipts remain preserved privately. No vendor runtime
source or compiled program is republished.

The next gate is an isolated, compiler-produced FNP-to-CNP communication truth
table with known nonzero and tie cases, followed by a native multiclass learning
check with an explicit field-by-field adapter contract. See `resident/NEXT.md`.
Do not begin with the six-bank repair experiment or use its forecast as a result.

Recount: `python -m resident.archive audit-public`.
