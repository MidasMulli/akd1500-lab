# Evidence map and next hardware questions

| Layer | Evidence in this repository | Open boundary |
|---|---|---|
| Host/PCIe | Bounded known-aperture adapter, MMIO counts, input/submission/fetch timing | DMA versus copied-output alternatives; interrupt/polling tradeoffs |
| Queue | Preserved full-ring publication failure; corrected eight-job/nine-slot burst with exact outputs/RAM | Longer-duration ring wrap, asynchronous producers, fine-grained cycle timing |
| FNP3 | Exact outputs and complete learning RAM through 512 neurons; reconstructed selector in CPU and Python | Larger connection budgets, input sparsity, near-capacity overlap and interference |
| Mesh | Passing 27/29/31/32-NPU single-pass configurations | Peak useful throughput, contention, bandwidth and memory saturation |
| Multi-learner | One plastic destination in this study; fixed CNP routing before it | Many simultaneously plastic banks, independent streams and labels |
| FNP2 | Compiler selects external-memory records for larger cells | Decoder, verified transport and matching behavioral model before execution |
| Persistence | Continuous learned state within a running native stream | Complete hidden-state capture, resumed plasticity and power-cycle durability |
| Power | Recorder driver, firmware, capture, calibration gates and synthetic tests | Assemble/interpose/calibrate hardware; matched card and host energy |
| Firmware/silicon | Known public runtime/register interfaces | Internal microarchitecture, firmware state, electrical spike/current traces |

All 32 NPUs participating is a coverage result. It does not establish maximum
clock, power, SRAM occupancy, bandwidth or arithmetic utilization. Silicon and
firmware ownership here means progressively observable and testable boundaries,
not a claim that inaccessible internals have been enumerated.

The next efficiency experiment should reduce unnecessary host transfers while
preserving a genuinely useful output, then vary width, sparsity, number of
resident contexts and event/update rates. Use optimized CPU task alternatives
as well as the exact neural-kernel reference. Include host CPU time and matched
energy domains; keep any simulator-only forecast separate from physical results.

For the power rig, confirm the actual module/interposer pinout, fit the sensor
on the complete card supply branch, and pass known-load calibration first.
No electrical ratio should be published before that gate.
