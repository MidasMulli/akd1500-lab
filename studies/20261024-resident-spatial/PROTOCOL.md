# Resident spatial fabric and native learning — frozen stage gates

New direct user authorization; previous studies remain closed. Maximum 24 physical sessions / 2,048 raw-image forwards, 120 seconds per owned process, two-hour ceiling in AUTHORITY.json. Stop on first failed audit; preserve all failed attempts. Corrections require a prospectively identified new attempt and freeze. Lock/fuser check before physical access. Only terminate owned PIDs. No firmware, voltage, clock or deployment modifications.

## Workload and scope

32×32 RGB -> HRC 32 channels, 8-bit 3×3 stride 2 -> CNP spatial 4-bit 3×3 stride 2, max pool 2 -> CNP 2-bit 1×1 projection with eight outputs -> FNP3 128-input, 128-record binary native learner with 32 selected connections per record. Hardware routing and one pass required. FNP2/registered host DDR weights rejected. Read every fixed CNP weight/parameter packet before and after, and complete learner memory after every event. Memory geometry and returned potentials are measured; physical die layout and individual spike times are not observed. HRC state is not covered by the CNP configuration readback routine.

Filters 8 -> 128 -> 256 -> 512, seed 107501; seeds 107601/107701 only at 512 after the first seed passes. Population and learning parameters stay fixed. Initial mapping and intermediate features must exactly match independent NumPy arithmetic. Compiler-only attempts 01 (pool directly to FC) and 02 (4-bit wide projection) failed and are preserved; attempt03 uses the exact same -1/0/+1 projection coefficients encoded in two bits. This correction changes the physical encoding, not the reference arithmetic.

## Stream and matched controls

64 events: 8 initial query cues, 32 teacher-labelled native updates (four consecutive presentations per cue), 16 repeated queries with teaching disabled, 8 novel queries with teaching disabled. Alternate cue labels are synthetic. Frozen twin starts from identical weights and receives identical cues but no teaching. This is a mechanism and resource test, not an application, generalization benchmark or autonomous discovery claim. Native learning parameters: initial plasticity .5, minimum .25, decay .125, competition .25.

CPU controls execute the whole pipeline and exact connection-selection rule in optimized C++ with NEON, preallocated convolutions, dense and zero-skipping single-core paths, plus a four-core OpenMP zero-skipping path. Five fresh-state CPU repeats. Verify all feature bits, scores and complete per-step RAM. Pin single-core/native controller to core2 and four-core control to cores2–5. Process CPU time includes worker threads. These are substantial exact controls, not proof of the globally fastest possible CPU implementation. Native transfer, enqueue and output fetch included in latency; observer readbacks/reference verification measured separately. No energy or calibrated power claim. CPU libraries/alternative kernels and thermal/order effects remain limitations until separately tested.

## Admissions

For each cell both arms, all three CPU controls must pass. Then cell8 seed107501 physical 8-query smoke, 40-event learn/check, and 64-event learning and frozen arms. Each larger width requires both prior width arms to pass the saved-array audit; only then run 64 events/arm. Extra seeds require the 512-filter seed107501 result first. Up to two temporal repeats of a passing full cell permitted; no automatic expansion on a failed gate. Before selecting a next stage, inspect the current saved audit.

Every query after event40 must preserve learned RAM exactly; frozen arm must never change it. Learning arm must show actual connection changes. Compare every native score and complete RAM snapshot with all CPU paths, and fixed physical memory with compiled packets. Per-event effects and performance endpoints kept separate. Store raw inputs, scores, state snapshots, hashes, ownership and failures; visualize from the saved qualified traces only.
