# Native correction and bounded record reuse — October 25, 2026 UTC

**The revised circuit corrects changing associations while preserving two
unrelated original memories, using native connection updates and a fixed-size
eight-record bank.** It passes 12 alternating reversals across three initial
connection seeds. This is a bounded synthetic capability, not CPU superiority or
general learned memory management.

The prior [error-only study](../20261025-prediction-error/ASSESSMENT.md) passed
stationary learning but failed reversal and capacity. Its negative results remain
in the same public numeric archive. The revision was frozen before hardware.

## Measured results

| Same four-context stream, three initializations | Read-only probes after reversals | All queries including acquisition |
|---|---:|---:|
| Native error gate + retirement | **1,728/1,728** | **1,864/1,956** |
| Matched representation, retirement path removed | 1,001/1,728 | 1,091/1,956 |
| Same bootstrap, then learning frozen | 1,299/1,728 | 1,441/1,956 |
| CPU latest-observation dictionary | **1,728/1,728** | **1,872/1,956** |

Each initialization has 12 reversals and 48 sealed probes per reversal, on the
same fixed stream/order. These are three connection initializations, not three
independent environments. Keys 0/1 change outcome; keys 2/3 stay fixed. All
843/843 stable-context probes pass in every arm; the native arm's original stable
connection vectors also remain exactly unchanged. Native probes have no ambiguous
or missing predictions. Across the three full native traces, 488 stored binary
connections are replaced, each record retaining exactly 32 connections.

The no-retirement control scores 439/576 on seed 114101 and 281/576 on each other
seed. It actually passes the first seed's short initial reversal, 48/48. A single
short demonstration would therefore miss the distinction. The trained-frozen
control benefits whenever alternating rules return to its original mapping.

The first-seed severed-prediction control scores 37/48 after one reversal versus
48/48 intact, and has only 12/16 correct bootstrap revisits versus 16/16 intact.
Forcing the prediction to appear missing permits unnecessary negative teachers,
which can interfere. This is a causal control for the saved-prediction gate.

## What physically changes

The five sequential FNP3 groups are fixed comparison `[1,3,0]`, fixed gating
`[2,1,0]`, fixed binary expansion `[2,2,0]`, plastic memory `[2,3,0]` and fixed
relay `[3,1,0]`. Only eight of the memory group's 32 used records are plastic;
24 are fixed forwarding helpers. This diagnostic uses five of the chip's 32
NPUs and is not a concurrency or whole-chip utilization measurement.

Each plastic record has 25 shared and seven variable binary connections. The
shared coding keeps matching records eligible under the existing native learning
competition rule. When an external outcome contradicts the saved native query
prediction, two predetermined teaching roles run. Positive evidence targets the
actual class; negative evidence targets its complement. The negative role uses
six key features and a retirement sentinel. Native competition selects the
record and can exchange its seventh key connection for the sentinel. Ordinary
queries contain no sentinel, so the obsolete response stops meeting threshold 31.
Later contradictory experience can reuse the bounded bank through the same
native update mechanism. No new records are allocated, no thresholds reset and
no program reloaded within a session.

The host provides raw cue, actual outcome and fixed query/positive/negative role
clocks. It derives the complement class from the external fact, not from a decoded
prediction/error, and never selects a record or patches weights. Both teachers
reuse the saved *pre-update* query-output SRAM pointer directly; prediction bytes
84–85 survive phase/actual writes to bytes 64–71. The native fixed circuit computes
the admission decision. Outcomes are external teaching, not chip-generated facts.

This changes both feature coding and teaching roles from the failed error-only
study. Its 0/48 reversal result is historical context, not an isolated causal
comparison with this revision. The matched no-retirement/frozen controls provide
the appropriate comparison for the new circuit.

## Verification and limits

Sixteen qualified physical sessions and 7,468 native forwards in this study;
34 sessions/9,772 forwards including the error-only predecessor. Every returned
output, native input, descriptor, source-address lineage and learning-RAM boundary
matches independent Python bit-set and separate C++ packed replays. Fixed helper
records and four fixed groups remain unchanged. Queries and refused teachers do
not alter learning RAM. Native/open/closed fixed-state comparator checks cover
24 truth cases each; synthetic prediction writes occur only in those cells.

The numeric archive re-audits without hardware or licensed SDK. Six deliberate
semantic corruptions are rejected, including coupled source/descriptor changes.
[Complete stable-record verification](STABLE_RECORDS.json) also checks all 36
words of each unrelated record, beyond its connection vector. No physical
execution or compiler failures occurred in these two studies. An overstrong
archive-only equality assertion and an early closeout guard rejection are
preserved in [REPAIRS.json](REPAIRS.json); no data were changed or experiments
rerun to repair either. Both authorities are closed and the device is free.

The CPU dictionary also succeeds, using just four entries for this workload;
it outperforms native online acquisition slightly. This is not an energy,
latency, RAM-contention or substrate advantage measurement. The comparator,
shared feature coding, retirement sentinel, objective and teaching roles are
engineered. This does not qualify importance-aware eviction under many unrelated
inputs, arbitrary sensory generalization, power-loss persistence, autonomous
host-quiet scheduling or an emergent self-designed policy. Those remain separate
tests. The next useful gate is interference and selective retention with more
independent changing contexts, before allocating more of the chip.

```sh
python -m admission.correction_archive audit
```

[Numeric metadata](../../results/20261025-native-correction.json) ·
[compressed numeric captures](../../results/20261025-native-correction.npz) ·
[full audit](../../results/20261025-native-correction.audit.json) ·
[prospective protocol](PROTOCOL.md)


[1080p60 replay](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/native-memory-correction-1080p60.mp4) ·
[4K60 master](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/native-memory-correction-4k60.mp4) ·
[video decode QA](../../visual/native-memory-correction-video-qa.json)
