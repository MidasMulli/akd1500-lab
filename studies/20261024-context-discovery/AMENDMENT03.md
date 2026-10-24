# CPU admission verifier repair before hardware

First CPU-only one-cycle trace matched saved inputs, outputs, full RAM, potentials
and raw-symbol dictionary, but the audit prematurely required the final16-history
CPU endpoint at the one-cycle prefix. Transition windows from the old periodic
stream mean that endpoint belongs to the frozen12-cycle gate. Restrict that single
endpoint assertion to12 cycles, as intended in the original staged protocol. No
native fixture or circuit changes. Preserve cpu_s113101_native_c1 as a failed audit;
refreeze and rerun under an attempt suffix before any hardware admission.
