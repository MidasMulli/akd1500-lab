# Repeated native image-episode replacement

The bounded physical campaign closed at 20:20:41 UTC on 2026-10-27. It passed its primary gates on three fixed image cohorts: 12/12 native replacements, 96/96 delayed recalls and 192/192 content/context checks. The same resident programme performs four consecutive replacement cycles per run, without reloading its weights between cycles. This is an exact image-episode memory mechanism, not a demonstrated general object recognizer, autonomous victim-search system or CPU efficiency advantage.

In practical terms, the chip holds eight image descriptors. Its learned control rule can make a matching old record reusable, allow a new useful observation to occupy it, and still recall the other records after distracting observations. We repeated this turnover four times without reloading the resident network. The host still presents the old candidate and teaches which context is useful.

## What changed

The previous study established one native conditional replacement on designed binary cues. This study drives the same separated control/content architecture with descriptors from recorded real-camera images, reverses the usefulness rule repeatedly, fills all eight requested memories before replacement, inserts 16 other image observations before each delayed exam, and protects the other seven records throughout every cycle.

Control remains adaptable while content is stable: the control learner keeps its normal threshold cap24; the content learner uses cap30. A native-authorized retirement changes the selected content record's local threshold to1. These are learning-eligibility thresholds, separate from the content readout threshold31 (an exact32-connection match).

This allows the native circuit to ask memory whether a presented episode is known before deciding whether to retire it. A known episode in a context learned as not useful can be retired; an unfamiliar episode in that context leaves content RAM alone. The native content pool then competes to learn a later useful episode in the reopened record. Host code never sends a physical neuron address or copies a recalled answer back into the feedback input.

The externally presented old episode still identifies the candidate. The device is not scanning all records to discover the least valuable victim. The usefulness learner is taught with explicit binary context labels, and the known-and-not-useful gate and six-phase schedule are designed.

## Inputs and what the chip receives

The input chain is: recorded CORe50 RGB128x128 images → our frozen host CNN's512 binary features →31 fixed balanced projections → constant-density64-feature encoding → native control/content network → eight native return bits.

The projection seed is27102770. Each of31 columns has256 positive and256 negative coefficients; the bit is1 when the projected score is positive. Each bit becomes a dual-rail pair. A canonical marker and alternate/swap marker complete64 features with32 active connections. Neither object IDs nor image-index labels enter this descriptor; object IDs are used only to select the prospective cohort and name the external image reference. All content teaching events use the same class label0.

The CNN and projection run on the host, not on the AKD in this study. There are zero encoder fits. The exported CNN SHA256 is9c61f5cf5d0e4734512e19106d74f3a9f9a2aebbc35c5a18fe1a4fd111e20d80. Integer and quantized-MPS features agree on all84 selected images. The source is the recorded CORe50 dataset, not a connected live camera. An exact image is queried after other experiences; unseen-angle/lighting generalization is not claimed.

Each cohort uses a seeded permutation of object IDs1..50, the first28 objects, frame80, with no outcome or geometry filtering. Sessions8,9,11 are three prospective recipe cohorts, not a claim that these captures have never been examined anywhere else in the project. Sampling seeds27102772/73/74 are fixed before screening; the procedural network initialization seed is27102771 in all three runs. There are three image cohorts, not three independent network initializations.

The closest pairs differ in7,4,2 of the31 projection bits for development, held1 and held2 respectively. This is diagnostic, not a selection criterion. For two canonical encodings separated by d projection bits, their overlap is32-d. Cap30 protects the observed nonidentical pairs with d>=2 while still admitting the one-marker retirement swap at overlap31. Closer collisions and arbitrary input geometries are not qualified by these results.

## Resident implementation

Six FNP3s are resident:
- (1,3,0): control learner,96 configured neurons, only0 and48 plastic;94 protected helpers/carriers.
- (2,1,0): fixed predicates.
- (2,2,0): fixed content projection and management-marker gates.
- (2,3,0): fixed native mailbox formatter.
- (3,1,0): eight single-class content neurons,256 binary-byte input slots,32 connections each.
- (3,2,0): fixed raw recall return.

Four of the six NPUs implement fixed support stages; the Explorer's3D scene focuses on the two learning groups. This is not all32 NPUs, a physical die layout or proof of simultaneous independent operation.

Two disjoint V1 chains communicate through card-side scratch memory, distinct from per-NPU learner RAM. A writes a256-word mailbox; B reads the first256 bytes and writes its header/eight returned words only into the64-byte reserved return area. A subsequently reads that same native buffer. Actual descriptor source/destination pointers and defined input bytes are audited. Host diagnostic reads are taken for verification, but those copies are not resubmitted as feedback.

Content input excludes utility/context, phase, key shadow, knownness and return-header fields. The deliberate canonical/swap marker remains inside the payload; separation does not mean absence of every management feature.

The host supplies observations, explicit context teachers, fixed role/learning-configuration clocks and completion polling. It supplies no learned retirement permission, neuron ID or direct RAM patch. The global retirement configuration is a fixed phase of every observation; only the native permission/input transform causes the appropriate record to update.

## Prospective stream and endpoints

Each full cell has1684 native forwards:
- Four initial control-teaching forwards.
- Eight initial episodes, four observations each, each observation using the same12-forward A/B protocol.
- Initial query of all eight episodes under both contexts.
- Four cycles: reverse the usefulness teachers; present old episode0,1,2,3 respectively in the now-not-useful context; present an unfamiliar not-useful cue; teach the next useful episode8,9,10,11; present16 other image observations from four distractor images; then query the expected eight records and check both contexts.
- Query phases disable learning and leave complete observable learner RAM unchanged.

The16 intervening observations are after new-episode acquisition and before delayed recall. They execute192 hardware forwards with learning phases enabled by the fixed protocol; native gating keeps these unknown not-useful inputs from changing the content bank. This is interference/retention under a running protocol, not an hours-long or power-loss persistence test.

The four original episodes4..7 remain protected throughout. A newly learned episode also becomes one of the protected records during later replacement cycles.

## Measured native results

| Image cohort | New record reuse | Delayed recall | Context checks | Initial full-pool checks | Retired physical records |
|---|---:|---:|---:|---:|---|
| session8 / seed27102772 |4/4|32/32|64/64|16/16|0,6,5,3|
| session9 / seed27102773 |4/4|32/32|64/64|16/16|0,2,3,4|
| session11 / seed27102774 |4/4|32/32|64/64|16/16|1,5,0,3|

Every defined native input, returned output and complete learner-RAM snapshot matches separate Python and compiled C++ exact-rule references. Actual feedback buffer addresses pass. All84 protected record/cycle comparisons are byte-identical over the entire associated cycle, not only its endpoint. Query-only content RAM and all204 unknown/not-useful observation protocols leave content RAM unchanged. The four fixed weight banks are checked at initial/final endpoints.

The different physical allocations follow native competition with different image inputs. The host never supplies those record IDs.

## Strong controls

All matched cells were separately executed on the same card and passed their raw audits.

| Arm | Correct initial checks | Successful targeted reuses | New episodes recalled | Delayed recalls |
|---|---:|---:|---:|---:|
| Learned manager |16/16|4/4|4/4|32/32|
| Trained policy then frozen |16/16|2/4|2/4|26/32|
| Retirement inhibited |16/16|0/4|0/4|22/32|
| Content frozen after fill |16/16|0/4|0/4|22/32|
| Normal cap, no capacity reset |16/16|0/4|0/4|22/32|
| Prior content cap24 |14/16|3/4 qualified victim reuses|4/4|32/32|

Retirement inhibition is a native input clamp, not a physically severed wire. The normal-cap control makes the same retirement connection edit but does not make its record reusable. Before the first new episode, that arm and the working manager differ at just one captured RAM word, word5: the selected record's local threshold1 versus30. All weights, swap budgets and other RAM agree.

The prior-cap24 comparison is especially important: final recall alone looks successful, but one initial requested episode was not recalled, so the first intended victim was not recognized. It fails the full-capacity replacement qualification. Its good final number is not evidence of four valid full-pool replacements.

Two bounded adaptive task-level CPU controls—exact associative/context memory and nearest-Hamming memory—also achieve32/32 delayed recall and4/4 new episodes on each cohort. They have eight content records and two context records, and do not emulate neurons or the learning rule. FIFO8 and LRU8 retain7/32 and12/32 in the development stream, respectively; that demonstrates the value of usefulness-aware policy, not CPU inability. The explicit usefulness oracle also passes. This is not a latency, energy, concurrency or substrate-advantage benchmark.

## Preserved failures and accounting

Physical accounting:9 attempts,14,208 reserved/observed forwards. The one-cycle pilot contributes736; eight full cells contribute8x1684=13,472. The headline three-cohort endpoint totals exclude the pilot rather than double-counting its prefix. There are11 source qualifications, including the failed first compiled-CPU attempt. The frozen physical cap10 and source cap12 were not exhausted or raised.

The initial C++ port accidentally kept a literal1 in the normal threshold override. Its feedback trace failed before any hardware use. The failed source/CPU files are preserved; the corrected C++ run uses byte-identical scientific inputs. No data, seed, encoder, threshold or endpoint was retuned after the held results.

Media development preserves four failed page loads from escaped-newline code generation and a failed encoder start from a local download-anchor variable shadowing the Explorer argument. These are presentation failures, not physical experiments. They were corrected and Chromium/browser/frame/decode checks are recorded. Do not infer successful browser parsing from the earlier command-line syntax-check exit codes.

## Explorer and capture

The repeated-memory profile is read-only at port8898. Its six selectable arms are actual separate hardware recordings of the development cohort. The source images, complete record words, per-step connection differences and eight returned bits are inspectable. Before/after inspection and neuron selection use the existing approved Explorer layout.

Neuron cores use actual returned binary flags for the two tapped policy units and eight content units. Helpers have no direct returned tap. Input pins show captured defined inputs; overlap scores are independently reconstructed from pre-update RAM, not measured electrical potentials. Spatial positions, camera motion and routing annotation are schematic. No electrical spike timing is measured.

The gallery's thumbnails are host image references whose descriptors exactly match captured connection masks. The AKD does not reconstruct or store RGB pixels. The native return precedes that step's update; the gallery can display before or after RAM. The capture is selected recorded events at60fps editorial cadence, not device throughput or a live connection to the released card.

The archive includes owned tooling, frozen input/model data, raw native/CPU traces, control/qualification receipts, preserved failures, the Explorer data and a1080p preview. The separate4K master is provided alongside it; its hash is in MEDIA_QA.json. The proprietary SDK/runtime and linked actor binaries are not bundled. Nothing is published to GitHub or X by this task.
