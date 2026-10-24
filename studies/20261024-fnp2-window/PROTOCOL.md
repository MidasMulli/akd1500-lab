# Native class-window learning over one persistent DDR-backed model

Purpose: avoid the measured FNP2 record-base boundary without CPU connection
updates. Keep an11264-record/512-input/two-class model in one guarded host-DDR
record array. Full inference reads the entire array. For teaching, the host
selects one5632-record view using the supplied class label. The chip performs
winner selection and connection updates in that bounded view. This is explicit
host scheduling, not a fully autonomous loop or a CPU/energy advantage.

Use ordinary HardwareDevice.program_external_data with compiler-generated full
and tile configurations. Only the tile's serialized FNP2 track offset is
relocated to the existing class slice; no connection or learning-state word is
patched. All control/input/output scratch remains in known card SRAM so it
cannot collide with the separately mapped4MBhost-DDR data arena. Protect the
entire live record array against every HardwareDriver host write after its
single initial load. API learn-memory views must equal direct full/sliced RAM.
No installeddriver/firmware/clocks or unknown register fields changed.

Stages, inspect before advancing:
1. CPU/compiler-only: exact raw weight decoding, class tags, descriptor offsets,
   input/reference uniqueness, single-swap teaching and all placement bounds.
2. Zero-forward ownership/provenance gate: program full view and both class
   views, verify all records unchanged, registered addresses and no host writes.
3. Direct full-bank native teaching negative control: one sealed high-address
   target, full state captured and failure retained. This branch stops on
   mismatch. A known failure never qualifies an application.
4. Independent windowed frozen control and learning pilot using that identical
   input/initial state. Each teaching call is followed by full-model read-only
   recall; capture complete bank at every boundary. Every score and word must
   match the correct full-model C++ CPU rule and independent Python replay.
5. Only after pilot checks, expand to eight sealed teaching targets across both
   classes, including previously failing global indices, followed by repeat
   weight initializations107301 and107401. Initial107101 matches prior family.
   This qualifies selected addresses, not every allocated record or chip-wide
   learning. All teaching probes have exactly one missing and one novel bit,
   avoiding claims about unqualified cross-view hidden selector state.

C++ CPU frozen/learning controls use five fresh-state repeats. Native process
caps:120seconds, at most24teaching inputs and48forwards per fresh process;
study cap16owned sessions/256forwards, includes the failed direct branch and
zero-forward gate. Source/fixture/binary freeze before execution; never overwrite
run paths. Exclusive research lock and fuser-free admission,2second native call
deadlines, unprogram cleanup, parent handle-release verification. Failed positive
gates stop expansion; no CPU repair, silent retry, relaxed reference or label
selection from chip scores. No forward follows a corrupt state.

Scientific cutoff18:00UTC and hardstop18:05UTC2026-10-24. Latest direct usergo;
all previous studies remain closed. Preserve manual-placement invalid control,
zero-forward observation repair, standard-SDK and record-stride failures.
