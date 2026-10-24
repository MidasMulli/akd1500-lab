# Native carried context and repeated learning pass the bounded test

October 24, 2026. The two-NPU network uses eight class-addressed binary records and a
fixed 256-channel decoder. Four records recognize the current context symbols A–D;
four predict outcomes P–S from the retained context and the same following symbol X.
The seeded representations and initial associations are engineered. Native learning
changes the context-to-outcome association; it does not discover the state encoding.

## What is new in this test

The host supplies only the current symbol. The first 128 bytes of the actual native
input come directly from the preceding native output buffer. The host writes the
current 128-byte half in that buffer, enqueues its address and obtains the next output.
After a single zero-buffer initialization, there is no normal host history copy or
history reconstruction. Descriptor pointers, full input readbacks and preceding output
packets verify the dependency. The decoder produces 16 active context bits, and zeros
in the current-symbol half, on every valid decoded response.

Thus X by itself is identical across four contexts in each query variant, while the
complete input differs through chip-carried history. The model responds to all four
contexts, adapts to externally taught association changes and retains unrelated ones.
The state is explicitly stored in output SRAM; this does not establish intrinsic
continuous neuronal dynamics or resolve the separate hidden query-history mechanism.

## Frozen stages and endpoints

Seeds 112101, 112201, 112301, 112401, 112501. First seed: compiler/packed CPU gates, then
native e0,e1,e2,e8. Each completed stage is independently audited from saved arrays on
Mac before admitting the next size. Frozen, unlabelled and erased-history e8 controls
follow. Only after those controls pass do the other four seeds receive the same four
e8 arms. Each physical run loads the program once and unprograms its owned model.

There are 23 physical sessions and 5,400 verified forwards, with no failed physical or
compiler attempts in this study. Every forward has a complete captured learning-bank
boundary, actual input/output and descriptor. Scalar Python and packed C++ exact-rule
references agree with every captured native boundary. A source-bound portable archive
audits the saved numbers and rejects five semantically corrupted, validly rehashed
traces: returned code, learning mask, history pointer, teacher class and dwell.

Each main run has 264 events: 8 old probes followed by 8 epochs of 32 events. An epoch
contains two presentations of each of two changed contexts and three four-context
query sets: full X, incomplete X after 1 ms dwell and incomplete X after 20 ms dwell.
The context precedes each X query. Teaching labels are external observed outcome
classes; the chip does not invent them. Two associations swap repeatedly between
two maps. There is no arbitrary grammar, long-history or unseen-context test.

| Across five initializations | Native | Frozen | Unlabelled | History erased at probes |
| --- | ---: | ---: | ---: | ---: |
| Changed-association checks | 240/240 | 120/240 | 120/240 | 0/240 |
| Unchanged-association checks | 240/240 | 240/240 | 240/240 | 0/240 |
| Context-code checks | 480/480 | 480/480 | 480/480 | 480/480 |
| Cumulative connection replacements | 1,280 | 0 | 0 | 1,280 |

The native arm passes 480/480 post-teaching predictions. Frozen and unlabelled each
pass 360/480 because two associations stay unchanged and the alternating schedule
revisits the old map. All arms retain their ordinary context-code responses. Counts
are repeated, correlated checks across five initializations, not 480 distinct facts.
Teacher-event responses precede the connection update and are not post-update scores.

The erased-history arm receives native teaching, then clears only the first 128 input
bytes before post-teaching X probes. Captured learning RAM at all 192 probe events per
seed is identical to the intact native arm. It returns no response at all 480 X probes,
while all 480 ordinary context-code probes still pass. No unrelated network fault or
changed visible learned mask is needed to explain the loss: removing the carried
context removes the information that disambiguates X in this constructed network.

## CPU comparison and operational scope

The strong CPU control identifies actual current cues from their bits, retains a
context byte and maintains a four-entry destination table. It receives the same
teacher information and passes 480/480. It is allowed its own efficient representation.
A fixed latest-symbol-only comparator passes 120/480; this is a memory ablation, not
evidence that a CPU cannot do the task. No native latency, CPU occupancy under load,
or electrical energy comparison was measured. No substrate advantage is established.

Learning competition 0.75 and the existing native learning rule are inherited from
the preceding qualified study. Initial plasticity 0.5, minimum 0.25, decay 0.125 and
activation threshold 24 stay fixed. Visible learned thresholds and swap budgets evolve
natively. There are no weight restores or threshold resets during a resident run.
All programs use learner FNP3[1,3,0] and decoder FNP3[2,1,0]; this is not all-chip scale.

The host schedules every symbol and supplies outcomes. It sleeps during the prescribed
dwell, then supplies X; no indefinite free-running operation or whole-Nano quietness
is claimed. Diagnostic reads occur after every forward. The film is recorded replay,
not live electrical activity. Selected connections and returned code values are measured;
class responses are decoded through the fixed readout. Geometry, camera, particles,
aggregated decoder links and playback expansion are schematic.

This qualifies a useful building block: a continuing, resident chip program can use
its preceding native state to interpret the same current stimulus differently, and
native teaching can change those predictions repeatedly. It does not yet establish
learned context discovery, autonomous scheduling, physical-world feedback, power-loss
durability, efficiency, or resolution of the older FNP2/multi-pass failures.
