# Mixed handoff collapses channels; bitwidth candidate fails

Closed October 23, 2026 Central. Thirteen physical sessions completed **824 fixed
inference forwards**. No learning or custom RAM adapter was used. Captured fixed
parameter RAM matches the original compiler packets before and after every run.
All virtual endpoint checks pass; native semantics differ by graph.

| Endpoint graph | Development vectors exact | Observation |
|---|---:|---|
|Terminal FNP source|128/128|Preserves the intended 64-bit vector|
|All-FNP identity|128/128|Preserves the vector|
|Directly fed CNP identity|128/128|Preserves the vector|
|Mixed FNP → CNP identity|2/128|Every nonempty vector becomes channel 63; empty stays empty|
|Directly fed CNP comparison|128/128|Intended pairwise comparison|
|Mixed FNP → CNP comparison|18/128|128/128 outputs match comparison of the collapsed-channel vector|

The frozen development bank contains all 64 basis vectors, zero, all-one,
alternating and seeded random patterns (seed 100101). Full mixed-vector
characterization was prospectively admitted after the eight-vector diagnostic;
semantic mismatch was a measurement endpoint, while transport, shape, memory
or ownership faults remained immediate stop conditions. This differs explicitly
from the earlier first-counterexample bridge study.

A prospective candidate changes only the declared source activation bitwidth
from one to two while preserving mathematical binary values, weights and
thresholds. Its parameter RAM is identical; configuration packets differ.
Native development smoke fails **0/8**, with the same collapse. No sealed native
candidate, comparator expansion or learning adapter was admitted. A four-bit
variant remains compiler-only and was not used as a physical fallback.

The collapse model explains the measured endpoint vectors; it is not a physical
internal-wire capture or unique diagnosis of firmware, compiler or silicon.
This study does not show that every mixed graph fails. The separate FNP-only
three-memory experiment is a different architecture, not a correction here.

The archive also retains the seven-cell single-layer compiler probe, with its
four FNP2-policy rejections. That probe made zero physical forwards and does
not establish hardware capacity. Original failed attempts and execution-source
hashes remain preserved. No CPU or energy advantage is established.

```bash
python -m resident.contract_archive audit-public
python -m unittest discover -s tests -p 'test_handoff_contract.py' -v
```

The public numeric archive includes all vectors and fixed-memory captures;
vendor programs/runtime remain private. No owned jobs remain; the device is free.
