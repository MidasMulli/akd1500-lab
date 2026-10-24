# Native execution characterization

The next-stage sparse-work and coupled-learning studies are now closed. They add
70 physical sessions, 19,712 forwards and 27,136 complete bank-state captures.
Initial-query service follows input/record size rather than selected connection
count in the tested FNP3 path. Two native banks can edit connections in the same
job and preserve their exact solo-learning state trajectories. See the
[sparse-work assessment](../studies/20261023-sparse-work/ASSESSMENT.md),
[coupled-learning assessment](../studies/20261023-learning-interference/ASSESSMENT.md)
and [source-bound chart](../visual/execution-limits-4k.png).

```bash
python -m metal.sparse audit-public
python -m metal.interference audit-public
python -m metal.cpu_compare audit-public
```

`sparse_prepare.py` uses the separately supplied compiler only. `sparse.py` owns
bounded admissions and exact per-step recounts. `interference.py` verifies the
four-mask state comparison; `cpu_compare.py` validates the contextual C++ CPU
implementation. Executed source/binary snapshots are retained privately. Final
export-completeness checks were strengthened after physical closeout; this is
not a native rerun. Generated programs and vendor runtime remain external.

This work tests execution-resource questions on the physical AKD1500 before
claiming useful CPU differentiation. The first campaign has 24 physical sessions,
9,568 forwards and a public numerical archive with a hardware-free recount.
A separate causal follow-up adds four sessions, 256 forwards and 40 physical
memory-region readbacks. Suppressing each compute NPU's thresholds removes only
its designated output group; both weight regions and the readout remain fixed.

* FNP3 population width adds service cost even with a fixed two-count output.
* CNP channel partitioning adds usable capacity across NPUs: 128 outputs/one CNP
  and 256 outputs/two CNPs have almost identical monitored service cost.
* Activation sensitivity differs between the tested fixed and plastic-capable
  paths. This does not establish sparse host ingress.
* Eight-input native teaching bursts pass both the normal setting and a bounded
  direct test of the existing pipeline helper. Register readback, all outputs and
  RAM at burst boundaries are retained. This is not proof of concurrent updates.

See the [assessment](../studies/20261023-metal-characterization/ASSESSMENT.md),
[prospective protocol](../studies/20261023-metal-characterization/PROTOCOL.md),
[queue addendum](../studies/20261023-metal-characterization/QUEUE_ADDENDUM.md),
[source-bound chart](../visual/native-execution-4k.png) and
[numerical metadata](../results/2026-10-23-metal.json).
The [causal ablation assessment](../studies/20261023-metal-ablation/ASSESSMENT.md)
separates actual memory readbacks and contribution tests from compiler mapping.
The [next test sequence](NEXT.md) targets the remaining execution, mesh and
plastic-state questions with discriminating controls.

## Evidence levels

| Question | Current experimental disposition | Next discriminator |
|---|---|---|
| Separate resident learning state | Exact RAM isolation established in the earlier relay study | Mixed-bank load, different teachers and shared-resource stress |
| Every configured neuron is a separate engine | Not supported by FNP3 width scaling | Input-word/connection-budget scaling and compact-output controls |
| Usable processing capacity across NPUs | Supported by output-exact CNP partition scaling | Placement/fanout/load sweeps; isolate mesh and readout bottlenecks |
| Only one input may be queued during learning | Contradicted in this runtime and bounded native bursts | More labels/topologies and sustained wrap; no universal override claim |
| Every neuron learns simultaneously | Not observed; at most one edited record per tested bank/event | Multiple independently driven plastic banks; distinguish queueing from execution |
| Exact per-neuron clock overlap | Not directly observed | Finer instrumentation or controlled bounds; do not infer an exact circuit from timings |
| Efficiency advantage over an effective CPU solution | Not established | Equal-quality, equal-deadline CPU comparison plus calibrated energy |

## Recount without hardware

```bash
python -m pip install numpy
python -m metal.audit_public
python -m metal.ablation audit-public
python -m unittest discover -s tests -p test_metal.py -v
```

The recount checks every native output, captured learning-RAM state, counter
delta, idle/configuration control, density grouping and matched teaching arm.
The chart generator additionally requires matplotlib:

```bash
python -m tools.visualize_metal
```

## Physical study tooling

`prepare.py`, `prepare_cnp.py` and `prepare_compact.py` use the separately
installed Akida SDK in compiler/virtual mode only. No physical device discovery
is performed during preparation. `src/metal_bench.cpp` uses the existing bounded
MMIO adapter and exact C++ rule, records native monitor counters and every
learning-RAM state, and compares serialized versus preloaded frozen bursts.
`src/queue_bench.cpp` tests a single known pipeline control in bounded teaching
bursts. The separately supplied vendor runtime and generated programs are not
redistributed. The public NPZ contains owned inputs, outputs, counters and RAM
observations, not executable vendor programs.

`run.py` and `queue.py` require an unexpired bounded authority, an explicit named
admission, an immutable source/fixture/binary freeze, an exclusive owner lock and
an unused physical-session/forward budget. This completed study has admissions
closed. It cannot be resumed by running the old commands.

All numbers are from one chip. DMA monitor cycles are not arithmetic-only cycles
or per-neuron timing. Buffer-end settings change the counter's scope; cross-mode
timing is not advertised as speedup. No CPU, energy, power-loss persistence or
whole-chip saturation advantage is claimed.
