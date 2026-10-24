# Native execution characterization, stage 1

Direct operator instruction: resolve the core questions experimentally; do not
treat documentation or undocumented behavior as silicon proof. This is hardware
characterization supporting a future CPU-differentiation experiment, not another
application efficacy demo. No energy advantage or physical neuron layout claim.

## Questions and discriminating controls

1. **What does the available device counter measure?** Compare repeated idle
   reads separated by 5 and 50 ms, event deltas, frozen configuration-counter
   deltas and repeated identical workloads. A host delay must not silently be
   interpreted as neural work. Counter scope is DMA-path occupancy; it is not
   initially assumed to be arithmetic-only or per-NPU timing.
2. **How does one NPU service neuron records?** Fixed 512 binary inputs, 32
   selected connections per neuron; widths 8/32/64/128/256/512 at the same NPU.
   Identical streams, input densities 0/1/8/32/128/256/512, three blocks of 16
   events per density. Width/activity scaling can distinguish constant cost,
   width-proportional service and stepwise capacity; it cannot uniquely reveal
   transistor-level execution without additional observability.
3. **What changes during learning?** After the inference prefix, 64 teaching
   events and 16 frozen queries. Capture every complete learning-RAM state and
   output. Count edited neuron records and compare to the independently coded
   rule. Do not assume the documented one-winner behavior or independent updates.
4. **What can be queued while learning?** Record effective batch-size requests
   through the existing runtime. Treat an enforced limit as an interface result,
   not proof that the physical chip cannot support more overlap.
5. **Is useful overlap observable?** For frozen weights, compare the exact same
   eight inputs submitted serially and as a preloaded resident burst with a spare
   queue slot. Record device counters and host completion separately. If an
   independent-branch graph compiles and passes virtual checks, admit its minimal
   output-exact smoke before comparisons. Compiler rejection is preserved and is
   not classified as silicon incapability.

## Stages and admission

* Compiler/virtual only: six width fixtures and A/B/AB branch feasibility.
* Instrument smoke: width 64, 16 initial inference + 8 teaching + 8 retained
  queries, idle controls and paired serialized/burst suffix.
* Inspect raw outputs, RAM and counter behavior before admitting any expansion.
* Conditional width screen: one pass per width. Stop a failed width, preserve it;
  inspect before a predeclared two additional temporal repeats. Total maximum
  physical sessions 24, forwards 30,000, each process 180 seconds.
* Branch experiments require fresh explicit admissions after their source,
  compiler graph and reference outputs are reviewed. No speculative register
  sweeping, clock/firmware/driver changes, unsafe addresses or physical wiring.

All exact outputs and complete learning RAM must match two implementations.
Counter data may qualify a timing observation only after the idle and repeated
workload controls are inspected. Non-additive timing by itself is not a literal
per-neuron concurrency trace. No host timestamps substituted for device cycles.

Keep failed attempts, exact source/fixture/binary hashes, owned job receipts and
admissions. The owner lock is held and other device users checked before opening.
Stop only owned processes at the saved cutoff. Close and release the device.
Any expansion beyond this envelope requires a new bounded study.
