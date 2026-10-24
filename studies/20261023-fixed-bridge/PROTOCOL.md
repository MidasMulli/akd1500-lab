# Isolated fixed FNP/CNP bridge

This separate inference-only diagnostic does not retry the failed learning adapter.
Freeze128 four-class support patterns from seed99901, with explicit one-class
winners, empty/equal ties and seeded random counts0..16. Lowest-index ties win.
Source bits are64 grouped neuron responses, with extra input zeros for the512-bit
FNP source. No labels, learning, custom RAM or intermediate host input.

Variants: all-FNP source/comparator/encoder/readout; direct-CNP comparator/encoder
with64-bit host input; mixed FNP source/CNP comparator/encoder/FNP readout. Final
output is four measured class-block counts,128 at the chosen class. Only ordinary
compiler programs. Reject multipass/external-memory/FNP2 substitutions.

Virtual truth tables and exact mappings before physical admission. In order:
all-FNP32 smoke then128 full; direct-CNP32 then128; mixed32 then128. Each requires
previous exact native output and unchanged fixed RAM. Stop at first mismatch;
save its output prefix and final fixed RAM. No retry/substitution in this study.
Maximum6 sessions/480 forwards;120s/process; own lock, known apertures, saved clocks.
Compiler and hardware outcomes remain distinct. No learning, throughput or energy
claim; this only isolates fixed categorical communication semantics.
