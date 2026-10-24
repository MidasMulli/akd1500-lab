# Native execution resources: measured rather than inferred from neuron count

**Stage 1 completed on one physical AKD1500.** Twenty-four physical sessions,
9,568 forwards, 969,312 native output values and 4,208 complete learning-RAM
snapshots were independently recounted. All physical sessions passed exact
numerical checks. A compiler-rejected branch merge is retained separately.
The study is closed, with no owned jobs and the device/owner lock released.

## Observations

### One FNP3's service cost grows with its configured neuron population

With 512 binary inputs, 32 selected connections per record, half-active inputs
and raw-score outputs:

| Neuron records in the same NPU | Median hardware monitor cycles/input |
|---:|---:|
| 8 | 3,909 |
| 32 | 4,389 |
| 64 | 5,039.5 |
| 128 | 6,309 |
| 256 | 8,869 |
| 512 | 13,989 |

The endpoint difference is 20 additional monitor cycles per added record.
The two extreme widths repeat in two additional fresh-program sessions with
the same medians. A fixed two-count readout control has 7,636 cycles at 64
records and 16,624 at 512: growing the population still adds service cost when
the final output size remains two values. The added readout NPU has its own
cost, so those absolute times are not the same graph as the raw-score path.

This is evidence of shared service capacity in this FNP3 configuration. It does
not identify the physical number of arithmetic lanes, the transistor layout,
or whether particular sub-operations overlap. It rejects the assumption that
adding configured neurons supplies proportionally more independent engines.

### Partitioning fixed work across NPUs gives usable capacity scaling

Compiler-generated CNP1 graphs retain the same 512-bit input, 32 selected
connections/output and two-count final readout. The first outputs' weights
are unchanged prefixes as more outputs are added.

| Convolution outputs | Compute CNP1 NPUs | Median monitor cycles/input |
|---:|---:|---:|
| 64 | 1 | 10,054 |
| 128 | 1 | 16,213 |
| 256 | 2 | 16,223.5 |
| 512 | 3 | 20,831 |

Each graph has one additional fixed FNP3 readout NPU. Doubling 128 to 256
outputs while adding a compute NPU costs only 0.065% more monitored cycles.
The 128- and 512-output cells reproduce their medians in two additional
fresh-program sessions. This supports useful parallel capacity across NPUs;
it is not a literal same-clock trace of individual neurons. No CPU comparator
or energy advantage follows from these kernel observations.

### Sparse activations affect the tested paths differently

The raw FNP3 64-record path is almost unchanged between zero and 256 active
input bits (5,033.5 versus 5,039.5 median cycles). The fixed CNP1 64-output path
changes from 3,929 at zero active bits to 10,054 at 256, and 16,782 at 512.
These are within-path observations. The kernels have different weights and
output contracts, so this is not a matched cross-path speed comparison.
All host input transfers remain dense; sparse host ingress is still unproven.

### The actual queue behavior exceeds the interface's one-input description

The runtime accepted a request for nine queue slots during learning. More
importantly, eight preloaded teaching inputs per released burst produced exact
sequential-rule outputs and complete learning RAM after each burst. This passed
for 64 inputs with alternating labels in each of two treatments:

* Normal learning mode: buffer-end selection OFF/output-end.
* Known pipeline helper enabled: buffer-end selection ON/input-end.

The actual event-DMA register changed from 925696 to 925952 in the ON treatment;
only the selected end-mode bits changed. The OFF and ON arms have identical
inputs, labels, initial RAM, all 4,096 output values and all eight captured RAM
boundaries. The host wrote no inputs or weights during a released burst.
The normal setting was restored and the device unprogrammed after each run.

This establishes bounded correctness beyond the interface description. It does
not prove overlapping synapse updates, independent simultaneous teachers, or
that the override is generally valid for other topologies and workloads.

Across ordinary serialized teaching events, each captured transition changes
at most one neuron record in its learning bank. Frozen queries change none.
That is a direct observation across these inputs and widths, not an exhaustive
proof about every possible program.

## Counter qualification and limits

The device monitor stays unchanged during idle waits of 5 and 50 ms. Its event
deltas are nonzero, configuration-counter deltas are zero during measured
forwards, and repeated frozen workloads are reproducible. A resident burst is
also checked against the same eight serialized inputs and outputs.

The monitor is on the DMA path. It includes interface/format/backpressure effects
and is not a per-NPU, arithmetic-only or electrical spike trace. Buffer-end mode
differs between inference and learning, and between the two override arms.
Therefore no cross-mode cycle ratio is presented as a learning speedup.
Hardware-counter and host elapsed time remain separate arrays.

## Preserved failure and next depth

Two separately compiled FNP3 branches pass individual virtual/compiler checks,
but a FullyConnected merge with two inbound layers is rejected by the SDK.
That is an interface finding, not proof the silicon cannot route such a graph.
The separately generated channel-partitioned CNP graphs provide a working path
for testing distributed fixed processing.

Remaining layers have concrete next tests: vary input-word and connection budgets
with a fixed compact output; separate arithmetic from shared mesh/backpressure
by changing placements and fanout; test independently driven plastic banks under
mixed teaching/query load; probe complete learning-state restart by exact suffix
replay; and qualify sparse ingress and calibrated card/host power. Do not replace
those questions with another small game or association demonstration.
