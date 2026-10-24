# Repeated native adaptation passes; query-history effect qualified narrowly

October24,2026. One resident FNP3 learner plus a fixed FNP3 decoder. Eight binary
records, eight externally addressed destination classes, one prototype/class.
Initial conditions, seeds, stages, budgets and stop rules were fixed before admission;
two prospective amendments qualify a failed CPU expectation and a selector control.

## Primary result

Each session programs the model once, exposes eight old examples/queries, then
performs twelve map-change epochs. Each epoch contains a prediction from the carried
native state, four fresh examples (two contexts twice), eight matched partial-cue
queries and sixteen native output-to-input predictions. Two synthetic permutations
alternate by swapping the same two destinations. The teacher stream supplies observed
context/destination pairs independently of the carried trajectory. No unseen-world
discovery, physical environment feedback or autonomous teacher generation is tested.

The last pre-teaching output stays in device SRAM across the observation/probe jobs;
the next feedback burst reads it directly. Every dependency pointer, input readback,
output packet and captured learning-bank boundary independently checks. The host
chooses the finite operation schedule, while the native model generates the subsequent
feedback state values. It does not compute replacement weights or reseed the carried
trajectory. Observation, learning and feedback are interleaved, not simultaneous.

The inherited competition0.25 setting learns the first change but raises the learned
threshold to16. The reverse example shares16 selected features, so strict eligibility
fails. Changing only the fixed SDK competition setting to0.75 gives cap8 and permits
continued native updates. Initial plasticity0.5, minimum0.25 and decay0.125 remain
unchanged. All settings stay fixed throughout each resident session; thresholds and
swap budgets evolve natively. This is a configuration qualification, not a new kernel
or undocumented learning algorithm.

| Across five fixed initializations | Native0.75 | Inherited0.25 | Frozen | Teacher disconnected |
| --- | ---: | ---: | ---: | ---: |
| Changed-rule checks | 120/120 | 60/120 | 60/120 | 60/120 |
| Unchanged-association checks | 360/360 | 360/360 | 360/360 | 360/360 |
| Post-update feedback consistent with current rule | 960/960 | 888/960 | 840/960 | 840/960 |
| Cumulative connection replacements | 1,920 | 160 | 0 | 0 |

Seeds111101,111201,111301,111401,111501. Each varies feature placement, old cycle,
changed sources and partial cue masks. Each main native run makes384 cumulative
replacements over46 updating events. It sees48 labelled changed examples. Because
there are twelve alternating changes, the terminal weight mask returns to its
initial map; cumulative edits, rather than initial-to-final net distance, describe
the learning activity. The changed records' metadata finishes at threshold8 and
swap budget8. The other six records retain their visible state.

The120/360/960 checks are repeated, correlated observations across five initializations,
not independent statistical trials or120 distinct learned facts. Pre-update shock
predictions are recorded separately and excluded from the post-update feedback score.
The latter evaluates the next state given each arm's actual current input; it is not
physical closed-loop task success or a common ground-truth episode endpoint.

## Exact-state qualification and preserved failure

Stage order: inherited e1→e2; native e1→e2→e4→e12; twelve-change controls; four further
initializations. One native e2 attempt stops at event46 because its captured partial
edit differs from the candidate CPU model in two words. Its47 completed forwards
and all saved data are retained. The on-chip program is not repaired.

On the saved prefix, a CPU model that omits unlabelled-inference score accumulation
fails; advancing that reconstructed accumulator matches the observed boundaries.
The scalar/packed models are corrected and the native programs and initial memory
are verified byte-identical. A fresh e2, held-out e4/e12 suffixes and four additional
initializations then match the corrected model at every captured boundary. The25
earlier CPU forecast runs are archived as superseded, not silently reclassified as
valid exact-state evidence. A corrected model fitting the failed prefix alone would
not have been sufficient qualification.

Across this study: **29 valid physical sessions,7,971 verified forwards**, plus the
one preserved47-forward failed-prefix attempt. Total observed completed forwards:
**8,018**, across30 physical admissions. Each main twelve-change run has364 forwards
and184 captured learning-bank boundaries. Every serial event is captured; a quiet
burst is checked at its endpoint, not through invented per-step RAM snapshots.
Each16-step burst completes by the first wake after a5ms sleep, with zero driver
reads and one release write during that interval. This is a worker observation
bound, not chip latency or proof the whole Nano is quiet.

## Query-history control

After the primary native e12 passes, prescribed controls add1 or8 incomplete S0
queries before the second change. Both preserve the visible learning RAM before
teaching; eight queries produce a different next partial edit. A further prospective
control replaces those eight incomplete cues with eight full S0 cues, holding
descriptor count and returned query codes constant. It also preserves visible RAM.

| Extra unlabelled queries | Visible learning RAM before update | Next partial mask versus zero-query case |
| --- | --- | ---: |
| None | Reference | 0 differing bits |
| One incomplete cue | Identical | 0 differing bits |
| Eight incomplete cues | Identical | 6 differing bits |
| Eight full cues | Identical | 0 differing bits |

The eight-incomplete case repeats exactly. Both eight-query cases return the same
eight codes, while their input details differ. This rules out descriptor count alone
as the explanation under this setup and demonstrates input-dependent query history
affecting a future native edit despite identical captured learning RAM. The reconstructed
score-accumulation model predicts all these masks. A specific hidden register is not
directly measured, alternative internal implementations are not exhaustively excluded,
and restore/reboot/power-loss persistence remains unqualified. Scope: one initialization
and these prescribed query prefixes.

## CPU control and practical interpretation

A source-keyed CPU transition table receives the same externally supplied information
and passes every native-arm matched recall and post-update feedback endpoint. Its
intermediate responses differ from destination-addressed native memory. Measured CPU
loop times across five seeds are about1.166–1.171µs/event for packed exact replay and
0.326–0.336µs/event for the task table, including dense code handling and checksum
work. These are CPU control costs, not a matched native latency or energy study.
No computational, speed or electrical advantage is established.

The demonstrated milestone is a bounded, host-paced resident adaptation loop:
fresh external examples update native memory; that memory changes subsequent
chip-fed predictions; unrelated associations survive repeated changes. Native
learning can remain useful across multiple changes without resets in these conditions.
It is not an indefinitely running or self-teaching chip, and remains a small two-NPU
experiment. Older high-address FNP2 learning and hardware-generated multi-pass replay
failures are unaffected.

Portable replay: `python -m plasticity.archive audit`. Numeric archive, source freezes,
the superseded predictions and selector controls are preserved in the result package.
Film and selector figure show recorded codes/selected SRAM, with schematic geometry,
camera, particles and expanded playback. No electrical neuron-spike trace is available.
