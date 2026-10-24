# Causal contribution of all 11 compute CNP parameter regions

Closed October 24, 2026 Central. Twelve sessions complete 1,056 exact native
forwards: a fresh baseline and 11 output-suppression treatments. All 23 parameter
packets per graph are read back exactly before and after execution, for 552 full
fixed-region readbacks. Every output, serial suffix and queued suffix matches
independent integer reconstruction.

The baseline program SHA matches the earlier ordinary compiler 2048-output
program exactly. Differential compilation changes thresholds 15→32 for one
contiguous output block, changing words in just one distinct CNP parameter region.
Native readbacks confirm that localization. With every input bit active,
unsuppressed baseline output is [1024,1024]; measured output drops are:

|CNP coordinate|Suppressed output block|Measured two-count drop|
|---|---|---|
|(1, 2, 1)|[0,187)|[187, 0]|
|(1, 2, 2)|[187,374)|[187, 0]|
|(1, 2, 3)|[374,561)|[187, 0]|
|(1, 3, 1)|[561,748)|[187, 0]|
|(1, 3, 2)|[748,935)|[187, 0]|
|(1, 3, 3)|[935,1122)|[89, 98]|
|(2, 1, 1)|[1122,1309)|[0, 187]|
|(2, 1, 2)|[1309,1496)|[0, 187]|
|(2, 1, 3)|[1496,1683)|[0, 187]|
|(2, 2, 1)|[1683,1870)|[0, 187]|
|(2, 2, 2)|[1870,2048)|[0, 178]|

The block crossing the output-group boundary correctly removes 89 from one count
and 98 from the other. All 11 distinct compute regions have an observed causal
relationship to graph output. The twelfth mapped NPU is the fixed FNP readout;
it is verified as part of the graph but is not separately ablated in this study.

This changes neuron-output thresholds, not physical NPU power. It does not trace
simultaneous arithmetic clocks, prove a transistor layout, or establish a CPU or
energy advantage. Other NPUs' parameter bytes remain unchanged. The intervention
supports the mapped regions' contribution; the scaling experiment separately
measures aggregate service behavior.

Compiler-localized block lengths 187×10+178=2048 were prospectively hypothesized
from parameter-packet lengths, then checked by exact per-word differences before
hardware access. There was no offset search or physical format sweep. Native
baseline and first suppression passed before the other 10 were admitted.
One chip and one frozen synthetic workload; no failed treatment or owned job
remains. Vendor programs/runtime are external.

```bash
python -m capability.contribution_archive audit-public
```
