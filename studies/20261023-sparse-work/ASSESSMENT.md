# Sparse connections did not reduce initial-query service in this FNP3 screen

Measured on one AKD1500: 54 native sessions, 9,696 forwards, 7,104 complete per-event
learning-RAM captures. All output values, teaching states, retained queries,
serialized/burst suffixes and counter controls passed C++ checks and an independent
Python integer recount. All36 compiler/virtual cells passed; no kernel or placement
substitution was required. Previous studies were not reopened.

## Initial inference

The same deterministic half-active input stream was used at each input dimension.
Connections per record were nested8/32/64. The learning bank stayed at FNP3 (1,3,0).
Compact outputs added FNP3 (3,3,0), with its initial/final RAM unchanged.

| Input bits | Records | Raw-score cycles | Two-count cycles |
|---|---:|---:|---:|
|128|64|3669|6102|
|512|64|5041|7641|
|1024|64|7078|9690|
|128|256|5973|7650|
|512|256|8869|11493|
|1024|256|13990|16614|

These are medians over 96 initial-inference events at 32 connections/record.
Changing that budget to 8 or64 left those initial medians unchanged, except the
256-record/128-input/8-connection compact cell:7654.5 rather than 7650 cycles.
The six64-record anchor cells each had two additional fresh-program temporal
repetitions, which reproduced their initial medians exactly. Other cells are
single-session screens with repeated input events, not three-session estimates.

With the two-count output held fixed,64→256 records added8.0625,20.0625 and36.0625
monitor cycles per record at 128/512/1024 inputs. The corresponding stored record
sizes are8/20/36 words, including four auxiliary words. This is consistent with
record scanning in this configuration; it does not identify the exact circuit
or physical lane count. The small-input raw-score path also shows additional
output-dependent cost, so raw output volume must not be treated as pure compute.

## Native teaching remains distinct

At128 inputs/64 records, raw-score teaching medians increased from 3838 cycles
(8 connections) to 4028(32) to 4517(64). Those figures use the teaching counter
scope and are not divided by inference timings. Complete state matched the rule
for every step. At most one record changed per bank per teaching event; queries
and the frozen suffix changed none.

## Limits and disposition

DMA monitor cycles measure buffer service, not arithmetic-only work. Dense host
transport remains in use. Compact output includes an added readout NPU, so it is
not an otherwise identical absolute speed comparison. No electrical energy,
useful-application CPU advantage, chip-wide peak or universal sparse-execution
claim follows. Closed admissions; no owned native job remains.

The source-transfer working-directory failure was preserved before any compiler
or device access. Final export/treatment completeness checks were hardened after
closeout without rerunning hardware. The exact executed sources and binaries are
retained privately in the hashed physical-source snapshot; generated/vendor
programs are not published. Public code includes the stronger final-export gate.

Recount: `python -m metal.sparse audit-public`. Numerical archive:
`results/2026-10-23-sparse-work.{json,npz}`. Source-bound chart:
`visual/execution-limits-4k.png`.
