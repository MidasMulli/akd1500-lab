# Evidence map and next hardware questions

| Layer | Evidence in this repository | Open boundary |
|---|---|---|
| Host/PCIe | Bounded known-aperture adapter, MMIO counts, input/submission/fetch timing | Confirmed same-score kernel DMA lowers caller CPU; full system CPU/IRQ and card energy remain |
| Queue | Preserved full-ring publication failure; corrected eight-job/nine-slot burst with exact outputs/RAM | Longer-duration ring wrap, asynchronous producers, fine-grained cycle timing |
| FNP3 | Exact outputs and complete learning RAM through 512 neurons; reconstructed selector in CPU and Python | Larger connection budgets, input sparsity, near-capacity overlap and interference |
| Mesh | Passing 27/29/31/32-NPU single-pass configurations | Peak useful throughput, contention, bandwidth and memory saturation |
| Multi-learner | Three connected 64-neuron plastic banks with exact per-step RAM checks | Independent streams/labels, CNP-interleaved failure root cause, useful task training |
| Compact readout | Matched threshold masks and a fixed second-NPU two-count reducer; identical learning state | Sparse/event input adapters failed; compiler-supported ingress proof and task-specialized CPU alternatives |
| FNP2 | Compiler selects external-memory records for larger cells | Decoder, verified transport and matching behavioral model before execution |
| Persistence | Continuous learned state within a running native stream | Complete hidden-state capture, resumed plasticity and power-cycle durability |
| Power | Recorder driver, firmware, capture, calibration gates and synthetic tests | Assemble/interpose/calibrate hardware; matched card and host energy |
| Firmware/silicon | Known public runtime/register interfaces | Internal microarchitecture, firmware state, electrical spike/current traces |

All 32 NPUs participating is a coverage result. It does not establish maximum
clock, power, SRAM occupancy, bandwidth or arithmetic utilization. Silicon and
firmware ownership here means progressively observable and testable boundaries,
not a claim that inaccessible internals have been enumerated.

The first compact-readout experiment reduced host latency from about 962 to
74 µs by narrowing outputs to two native group-support counts. The matched CPU
still took about 18 µs. Neither that result nor the routing mesh establishes an
AKD substrate advantage. Threshold counts are synthetic detector-bank outputs;
their application relevance is still a separate question.

The next efficiency experiment should test sparse/event input and task-relevant
sufficient outputs, then vary width, sparsity, number of resident contexts and
event/update rates. Use optimized CPU task alternatives
as well as the exact neural-kernel reference. Include host CPU time and matched
energy domains; keep any simulator-only forecast separate from physical results.

For the power rig, confirm the actual module/interposer pinout, fit the sensor
on the complete card supply branch, and pass known-load calibration first.
No electrical ratio should be published before that gate.

Balanced fixed 4+4 encoding prevented the three-bank response saturation seen
with positive-only fanout in 192 matched confirmation inputs. This supports a
better-controlled native mechanism, not useful task accuracy. The image-service
wait-path repair separately passed exact actions, outcomes and neural state
through real worker outages. Direct card power remains unmeasured.
