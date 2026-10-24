# Native partitioning before any further physical image run

The two image studies are closed. No further quality repair, exemplar shrink,
new model or test-seed replacement belongs to them.

The next design should keep the accuracy-qualified 3200 exemplars, inputs,
2×2 spatial-OR transform and decision fixed. It must first resolve representation
by the native program/compiler interface; a 3200-filter single CNP layer is not
accepted by the installed compiler.

1. Define two 1600-exemplar banks with explicit labels/biases and equal coverage.
   Prove exact output assembly against the existing 3200-count matrix on all
   frozen inputs. This is an interface/graph change, not a new classifier fit.
2. Inspect supported branching/concatenation/output contracts and native record
   routes. `fork_AB` in the metal study already failed because the merge FC
   requires exactly one inbound layer. Reuse that failure; do not assume a
   generic multi-input FC solves this. Source/virtual checks precede mapping.
3. Freeze either a compiler-supported single-pass graph or a separately reviewed
   disjoint-bank program construction. State whether bank selection/output
   collection is host-driven; do not call it autonomous native routing. Reject
   accidental external-memory, remapping-per-query or hidden multipass changes.
4. Only after exact virtual and mapping checks pass, create a new bounded
   physical protocol. Read every declared bank region before/after a 64-query
   smoke and check all counts/classes, nonzero suffixes, ownership and counters.
   Inspect before admitting the full frozen test and temporal repeats.
5. Measure total upload, submission, completion, result drainage and host decode.
   Report any host bank-control cost. CPU target: the actual compacted NEON
   path at 20.5445µs median, with matching input-processing boundaries. Remeasure
   controls interleaved with the admitted native design to control machine drift.
6. Use a smaller sufficient native readout only if its decision is independently
   exact or a new accuracy-bounded model is prospectively justified. Returning
   3200 counts is real transport cost; hiding it would invalidate the comparison.

No calibrated energy claim until the physical card-power rig is assembled and
checked. A successful mapping is not a scientific result; a whole-chip drawing
is not utilization. Preserve the compiler ceiling and failed small bank when
publishing the next stage.
