# Learned transitions: primary pass, hardware-generated replay held

October 24, 2026. New study, distinct from the earlier programmed feedback ring.
Stages were admitted sequentially after saved-array checks, with a frozen specification,
single-owner device lock, five fixed initializations and finite execution limits.

## What changed

The earlier ring changed selected connections without changing the sequence followed
by native and frozen arms. Here two externally taught transitions alter the native
sequence itself. Eight class-addressed FNP3 records recognize encoded current states;
a fixed second FNP3 maps the responding class into the next state code. The next
descriptor reads that code directly in device system SRAM. The host prepares the
finite schedule but does not choose subsequent states or compute weight repairs.

Each fresh session starts with an engineered old cycle and eight old examples,
then eight unlabelled partial-cue queries, two changed examples with swapped
destinations, then eight unlabelled queries. Teacher classes name observed destinations.
Learning remains enabled in the native and unlabelled arms. The frozen arm disables
learning before the changed examples. Query-only rollouts preserve visible learning
SRAM and the complete fixed decoder bank.

| Endpoint across five initializations | Native | Frozen | Teacher disconnected |
| --- | ---: | ---: | ---: |
| Changed associations recalled | 10/10 | 0/10 | 0/10 |
| Unchanged associations retained | 30/30 | 30/30 | 30/30 |
| Predicted states matching new graph | 1,280/1,280 | 309/1,280 | 309/1,280 |
| Connection replacements per complete run | 32 | 0 | 0 |

Seeds: 110101,110201,110301,110401,110501. Each initialization varies feature
placement, old cycle, changed sources and partial cue masks. Rollout cases are all
eight starting states ×32 steps. Correlated trajectory positions are not independent
statistical trials; the five controlled initializations, rather than a binomial
interval over 1,280 positions, define the replication scope.

The primary stage order was diagnostic-only →1→2→4→32 dependent steps for seed110101,
then matched controls, then four additional seeds. **19 valid physical sessions,
4,390 successfully verified forwards.** One failed frozen-control transport run
completed 26 diagnostic events and 64 rollout events before stopping. The separate
one-pass inference replay baseline adds eight forwards. Total observed completed
forwards including that invalid control: **4,488**. There were 23 scientific physical
admissions, plus a separate zero-forward owned cleanup.

Every primary returned output and complete diagnostic learning-bank capture matches
independent scalar Python and packed C++ replay. Each 32-step chain completes by the
first wake after a 5ms sleep, with zero driver reads and just one queue-release write
during that interval. This is an observation bound, not device latency or whole-host
silence. Raw SRAM packets and descriptor pointers bind output payloads to subsequent
inputs; no labels are delivered during rollout.

## CPU controls and interpretation

The exact native-rule CPU replay passes all visible states. A stronger task-specific
eight-entry transition table also passes old/post-adaptation recall and all rollout
endpoints. Its intermediate response between the two teaching events differs:
native destination-addressed reassignment temporarily removes the other association,
while a source-addressed CPU table preserves it until its own update. The first CPU
check incorrectly required those intermediate algorithms to agree and failed; that
failure and the corrected endpoint comparison are preserved.

Across the five native-arm CPU repeats, exact packed replay measures about
1.081–1.091µs/event, and the task table about0.361–0.372µs/event. These timed loops
include dense conversion/decode and checksum work; they are CPU control timings,
not calibrated energy measurements or a matched native device-latency comparison.
No computational, speed, energy or CPU-representability advantage is established.

## Preserved transport and replay failures

The first frozen 32-step control passes its first chain and fails its second.
Inspection shows the SDK's `toggle_learn(false)` re-enables pipelining after a
learning-SRAM readback. Explicitly disabling pipelining again repairs the dependent
transport; a fresh source freeze and repeated control pass precede expansion. The
failed trace is excluded from the scientific comparison.

Separate compiler-only replay screens preserve rejected two-NPU deep mappings,
insufficient selected FNP2 placements, and the `hwpr_loopitself` option rejecting
this IP target. None proves an absolute silicon limitation.

A second, inference-only adapter repeats the same complete compiler-generated pass
with the native-learned captured weights. One-pass baseline passes8/8. The first
two-pass install fails on duplicated tracked source addresses. Giving each pass
unique spans with identical bytes resolves that allocator error, but the next
attempt times out during multi-pass NPU configuration **before any forward**.
No four/eight-pass hardware expansion follows. A compile-only C++ type error is
preserved separately. The exact configuration-timeout cause is unresolved.

The failed owned installation was reset without inference. Saved post-reset values
show all three DMA run bits clear; the device has no process owner. No indefinite
replay, hardware-generated recurrence, continuous learning across reconfiguration,
or hidden-selector persistence is established.

## Remaining boundaries and next gate

This is a minimal eight-record associative transition test with an engineered prior,
one prototype per class, external examples and one map-change episode. It qualifies
changed native forecasts, retention and finite model-driven trajectory separately.
It is not planning/search, new-world discovery, arbitrary repeated reversal or an
end-to-end world model. The older high-address FNP2 learning failure is unaffected.

The next technical gates are a documented-and-measured finite replay configuration
contract, then repeated map changes without threshold resets. Those should precede
claims of self-sustaining learning or another application demonstration. A fully
quiet Nano and electrical energy remain separate unmeasured requirements.

Reproduce the portable raw audit with `python -m transition.archive audit`.
The public archive hash and all endpoint rows are in
`results/20261024-learned-transition.audit.json`. Recorded film geometry, class-flow
particles and timing are schematic; returned outputs and selected SRAM connections
are evidence-bound. No electrical spike tracing is depicted.
