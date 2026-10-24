# Evidence map and next hardware questions

**Current priority: native resource characterization before another application
demo.** Documentation is a hypothesis source. Resolve execution, queueing,
interference and counter scope experimentally, then admit a CPU-differentiation
workload only if a practical optimized CPU baseline is a meaningful comparator.
The first [metal campaign](metal/README.md) is complete: 24 sessions, 9,568
forwards, population-width and activity sweeps, CNP partition scaling, and direct
teaching-pipeline tests. Numerical archives and mutation-tested recounts are
public. Per-neuron clock overlap, independent simultaneous teachers, sparse
host ingress and electrical efficiency remain separate questions.
A separately bounded [causal ablation](studies/20261023-metal-ablation/ASSESSMENT.md)
adds four sessions: full physical parameter readbacks and selective suppression
confirm both compute NPUs contribute to their distinct output groups.

Two further closed studies added 70 sessions and 19,712 forwards. The
[sparse-work screen](studies/20261023-sparse-work/ASSESSMENT.md) separates input
size from selected connections: query service is consistent with record scanning,
while teaching cost responds to connection budget. The
[four-mask learning study](studies/20261023-learning-interference/ASSESSMENT.md)
shows both banks can edit connections in the same job while matching their entire
solo state trajectories across three seeds. Neither physical clock overlap nor
independent simultaneous teachers follows. The CPU contextual control reproduces
the same learning states; useful system differentiation remains unestablished.
The next discriminator is compute-versus-mesh capacity and genuinely independent
plastic input/teacher paths, followed by a qualified system comparison.

| Layer | Evidence in this repository | Open boundary |
|---|---|---|
| Native graph compaction | 16 sessions/2624 forwards; exact outputs/RAM;512-detector half-active service5.92% lower and captured parameter payload34.86% lower | Extra NPU and synthetic four-copy domain;256-detector and fully-active cases lose;21.05% burst gain is zero-activity only; CPU/energy advantage unestablished |
| Learning-policy screens | Equal-resource delayed-feedback policies, preserved failed repair and fresh four-seed validation | CPU forecasts only; quality/resource gates failed and no native learning run admitted; no internal-algorithm replacement |
| Native execution resources | Counter-qualified FNP3 width scaling; nearly constant CNP monitor cost when 128 outputs/one CNP grows to 256/two CNPs; exact teaching bursts with pipeline flag OFF and ON | Counter is DMA-path scope, not per-neuron timing; isolate input-word service, mesh contention, concurrent plastic teachers and energy |
| Host/PCIe | Bounded known-aperture adapter, MMIO counts, input/submission/fetch timing | Confirmed same-score kernel DMA lowers caller CPU; full system CPU/IRQ and card energy remain |
| Queue | Preserved full-ring publication failure; corrected eight-job/nine-slot burst with exact outputs/RAM | Longer-duration ring wrap, asynchronous producers, fine-grained cycle timing |
| FNP3 | Exact outputs and complete learning RAM through 512 neurons; reconstructed selector in CPU and Python | Larger connection budgets, input sparsity, near-capacity overlap and interference |
| Mesh | Passing 27/29/31/32-NPU single-pass configurations | Peak useful throughput, contention, bandwidth and memory saturation |
| Multi-learner | Three connected 64-neuron plastic banks; a separate five-NPU recognition/routing graph has two distinct phase-gated teachers and exact per-step RAM checks | Simultaneously arriving independent streams, richer branching/messages, CNP-interleaved failure root cause |
| Selective native repair | Three seeds: recognition-only updates restore 64/64 changed-signature destinations, retain 64/64 old responses, downstream RAM unchanged; frozen and routing-only 16/64 | Synthetic task on one chip; simple CPU also solves it. Test controlled communication interruption, real sensor streams, size/activity scaling and calibrated energy |
| Recovery | Bounded probing corrects stale failure memories during real image-worker recovery; 72/72 completed | Reactive also completes 72/72 at two fewer operations; test value under a preregistered cost/latency objective |
| Visual-action learning | Larger sealed bank: learned 551/640, frozen 534/640, constant 555/640; four of five pair wins, gate failed; all 1,280 selected outcomes freshly checked | Gain below preset threshold and conditional interval includes zero; useful visual/action generalization remains unqualified |
| Continuous decisions | Native queries, 24-frame commitments and outcome feedback; 115 decisions, 507 online replacements, all frames and RAM verified | First pair regressed: learned 1,418 px versus frozen 1,554; stop gate closed seed/test expansion. Hazard coverage and consequence objective need development |
| Changed physics prediction | Exact shared trained prefix; continued native learning 12/48 versus trained-frozen 4/48 changed-physics landing bands; all 4,140 forwards and RAM states recounted | Absolute accuracy fails 65% gate and best constant reaches 19/48. No expanded seeds or learned paddle control; isolate new-rule fitting, generalization and retention without rehearsal |
| Native relearning history | New-rule-only A 46/64 versus relearn 28/64 and frozen 12/64; B 47/64 without reminders; CPU task learner 44/64; all 6,752 forwards and RAM exact | One native seed. Learning history impairs this case; visible stale weights and hidden history are not isolated. The 32/48-connection CPU forecasts fail; no native expansion |
| Same-task CPU differentiation | 18 exact transport replays, 40,320 forwards; queued read-only DMA cuts caller CPU from 342.3 to 54.7 µs/input; exact CPU neural replay about 11.2 µs/input | Every tested AKD path remains more expensive on this recorded task. Need larger useful resident computation, compact decision readout and complete energy domains; no silicon-peak conclusion |
| Compact readout | Matched threshold masks and a fixed second-NPU two-count reducer; identical learning state | Sparse/event input adapters failed; compiler-supported ingress proof and task-specialized CPU alternatives |
| FNP2 | Compiler selects external-memory records for larger cells | Decoder, verified transport and matching behavioral model before execution |
| Persistence | Continuous learned state within a running native stream | Complete hidden-state capture, resumed plasticity and power-cycle durability |
| Power | Recorder driver, firmware, capture, calibration gates and synthetic tests | Assemble/interpose/calibrate hardware; matched card and host energy |
| Firmware/silicon | Known public runtime/register interfaces | Internal microarchitecture, firmware state, electrical spike/current traces |

All 32 NPUs participating is a coverage result. It does not establish maximum
clock, power, SRAM occupancy, bandwidth or arithmetic utilization. Silicon and
firmware ownership here means progressively observable and testable boundaries,
not a claim that inaccessible internals have been enumerated.

The signal-relay experiment connects isolated learning, resident weights and
native inter-NPU communication in a functional task. Each bank learns a
different association with its own teacher. A fixed categorical message stays
on-chip, allowing recognition to be repaired without updating downstream
routing. That is a bounded modular-learning result. Its successful small task
does not establish a substrate advantage: the software comparator also reaches
100%. Efficiency still needs enough useful resident computation, matched
accuracy/retention, optimized CPU alternatives and calibrated whole-system
energy. Merely adding replicated classifiers is not sufficient.

The larger Mario study added an observed frame-count cue after two RGB-only
encodings aliased identical playfield observations with opposite consequences.
That repaired the measured alias, but balanced training and the new cue did not
qualify a useful policy. A separate prospective confirmation held the resulting
pipeline fixed and still failed its learning-effect gate. More initializations
of this same pipeline are not the next priority. A new protocol should first
test whether local motion/hazard features and shorter, revisable motor commits
predict consequences reliably, against clock-only and constant controls. Seal
new encounter families only after that development gate passes.

The synchronized Mario capture samples real native match responses and actual
RAM changes while the game runs. It does not observe electrical neuron firing,
and the protected shadow queries do not perform frame-by-frame replanning.

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
