# CPU contextual control for the coupled-learning captures

CPU-only; no device discovery or physical-device open. Use an O3, native-CPU
compiled bitset/popcount implementation of the two observed native learning
rules and an equivalent direct categorical readout. Give it the exact saved
dense inputs, teachers and learning masks. Require every output and both final
complete bank RAM states to match the physical captures. One repeat of seed 97101
mask 11 is the smoke gate before 128 repeats of each of12 full native arms.

Each repeat recreates both banks and replays the 512-event prefix outside the
timed interval, then times128 test events with dense-input packing, both bank
computations, teaching and categorical/count output included. Bound each CPU
process to 30 seconds. Record wall and calling-thread time independently; retain
all numeric timings, inputs bound by native receipt, and host context.

This is a contextual CPU implementation, not a claim to CPU optimality or a
useful-application comparison. Native adapter latency includes transport and
50 us fetch sleeps; the CPU has no PCIe round trip. Native late confirmations
overlapped artifact handling. No clean whole-system CPU/IRQ or electrical energy
comparison is established, and no chip-level speed ratio is claimed.
