# Three plastic populations through native FNP OR encoders

This is a different architecture, not a fix or success for the failed FNP→CNP
handoff. Larger single native layers select FNP2 rather than distribute across
FNP3 in the compiler probe; no external-memory fallback is admitted here.
Use only the established FNP→FNP path. Three plastic banks with512 inputs each,
widths64/256/512,32 connections per record,4 classes,T27. Each fixed FNP encoder
ORs responses within a class and expands each present class into128 bits. Multiple
class groups may be present; it is not an argmax or an exclusive categorical code.
The final fixed readout returns four support counts. Six resident FNP3 resources.

Freeze the same sensory task generator and noise seeds as resident/design.py,
count3, seeds102101/102102/102103 for native initialization. Different per-bank
permutations and teacher labels; teacher/phase/domain not input features. Every
arm replays128 training events per bank from a fresh program, including hidden
selector history. Then old64, changed-before64, repair128, changed-after64 and
old-retention64. Repair enables only bank0; frozen enables none; wrong-bank
changes only bank2 using destination labels. No retention reminder teaching.

Before hardware: CPU task screen at all three widths on seed102101; require
old>=85%,changed-after>=85%,gainoverfrozen>=30points,retention>=80% and complete
protected downstream RAM unchanged. No task/threshold/learning-policy tuning.
Compiler single-pass/FNP3-only and complete virtual-graph checks must pass.

Native mechanics: width64 smoke, then256,then512. Each replays the full training
prefix plus8/8/16/8/8 test events. Require every output and every complete bank
RAM snapshot exact under the separately implemented rule; fixed memory unchanged.
Stop on mismatch; no smaller substitute following a full-confirmation failure.

At512, full repair/frozen/wrong-bank seed102101; inspect task and memory gate
before admitting full repair/frozen for reserved seeds102102/102103. Maximum10
sessions/9000forwards (outer23 cumulative including prior13);120s/process and saved
clock. Exclusive lock, owned cleanup and source/binary/fixture freezes. CPU
forecasts remain separate from hardware; logical records are not all changed
neurons. Record actual connection-edit events, changed records and fixed RAM.
No CPU speed/energy, general learning or full32-NPU plasticity claim.
