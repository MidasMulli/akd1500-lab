# Fixed components pass; their compiler-produced handoff fails

Closed October 23, 2026 Central. Five owned sessions completed **321 inference
forwards**. No learning, custom record adapter, teacher labels or auxiliary RAM
edits were used. All programs were unmodified single-pass compiler outputs. Every
captured fixed-memory region matched its compiler packet initially and finally.

| Compiler graph | Smoke | Full truth table | Physical result |
|---|---:|---:|---|
|FNP source → FNP compare → FNP encoder → FNP readout|32/32|128/128|Exact|
|64-bit host input → CNP compare → CNP encoder → FNP readout|32/32|128/128|Exact|
|FNP source → CNP compare → CNP encoder → FNP readout|0/1|Not admitted|Stopped on first output mismatch|

All three graphs passed the same 128-case virtual support-pattern truth table.
The mixed graph's first pattern activates 16 source channels in class 0. Expected
readout `[128,0,0,0]`; measured `[0,0,0,128]`. It was stopped after that first
counterexample; no sweep, retry, permutation repair or full mixed run followed.
Thus this is not a 128-case estimate of mixed-network accuracy.

This narrows the earlier learning-scale failure: an inference-only, ordinary
compiler-produced version of the mixed FNP→CNP graph also fails. Both component
paths work under their own input/output contracts. The composition requires
investigation of representation, coordinates, routing and compiler support.
The test does not identify a transistor defect, prove a particular firmware bug,
or establish that all FNP→CNP graphs are impossible. The source output/internal
message was not electrically probed. The whole graph's output and fixed parameter
memory were measured.

The older two-learning-bank FNP-only result remains valid. The new six-bank/17-NPU
mapping remains compiler-only; there is no new scaled learning, useful task,
CPU-efficiency or energy result. No further physical work is admitted in this
closed diagnostic.

Next: inspect the compiler's fixed source output and CNP input-coordinate contract
before another learning adapter edit. First reproduce the isolated failure using
an independently assembled supported bridge or explicit documented reshape, with
its own prospective controls. A manually discovered channel permutation would
require a separate sealed check; do not fit one and report its training cases.

Hardware-free recount:

```bash
python -m resident.bridge_audit
python -m resident.bridge audit-public
```

The public arrays contain the exact input patterns, output prefixes and fixed
memory readbacks. Proprietary runtime sources and compiled programs remain
private. No owned jobs remain; the device is released.
