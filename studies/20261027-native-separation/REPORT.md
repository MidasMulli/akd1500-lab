# Native separation and integrated memory manager

Scientific campaign closed 2026-10-27 at 19:01:42 UTC. This is a bounded hardware mechanism result, not a finished general-purpose memory service or a CPU performance advantage.

## Result in plain language

The AKD holds two separate learning groups. One learns whether a context is useful; the other holds eight content records. Changing the context rule does not damage recognition of the content. When an already stored cue is subsequently presented in a context now learned as not useful, the native controller permits retirement of that cue's record. A later useful cue occupies the reopened record while the other seven memories remain unchanged.

The host presents observations and explicit context teachers and advances a fixed schedule. It does not send a selected neuron address, compute the retirement permission, copy a recalled answer back as a new input, or patch the learner's RAM. The native rule that joins learned usefulness with recalled knownness is designed, not discovered. The presented old cue identifies the candidate; this prototype does not scan all records to find a globally least valuable memory.

Compared with the prior reclamation study, the host no longer supplies the retirement marker directly. A learned native policy, native recall and fixed native predicates produce it. Transient context and control fields are separated from the content learner's input.

## Staged evidence

1. A stock V1 two-input FullyConnected branch was rejected before physical use. We preserved that failure and built a different, explicitly qualified transport.
2. The first physical mailbox attempt failed after one returned forward because shared ingress retained the wrong role width. The repaired fixed role-switch protocol passed 64/64 native forwards, including exact device buffer addresses, return placement and untouched other bytes.
3. The primary integrated cell passed 546 native forwards and every output, defined input field and full learner-RAM comparison against independent Python and compiled C++ references.
4. Four matched physical controls each passed their raw audits and each failed the final new-cue recall, as prospectively expected.
5. Two further full native runs, under a second initialization-and-cue seed and two sealed presentation orders, passed the same endpoints. No further physical expansion was launched.

## Resident architecture

| Role | FNP3 coordinate | Function |
|---|---|---|
| Policy | (1,3,0) | 96-neuron group; only units 0 and 48 eligible for learning; 94 protected helpers/carriers |
| Predicates | (2,1,0) | Fixed usefulness/knownness/phase predicates |
| Projection | (2,2,0) | Fixed content isolation and management-marker gates |
| Formatter | (2,3,0) | Fixed 256-word native mailbox formatter |
| Content | (3,1,0) | Eight single-class content neurons; 256 binary input bytes; 32 connections each |
| Return | (3,2,0) | Fixed raw eight-word recall return |

Six resident FNP3s are used, not all 32 physical NPUs. These form two co-resident V1 chains advanced serially by the host. The disjoint chains and native mailbox are a custom deployment; this is not evidence of simultaneous independent execution or a free-running scheduler.

The mailbox is card-side scratch memory exposed through our bounded PCIe driver aperture (0x20000000..0x20100000), distinct from per-NPU learner RAM. We do not claim that this transport buffer is inside an individual NPU or establish its physical position on the silicon. A produces 256 raw uint32 words. B reads the first 256 bytes and writes only the 64-byte reserved return area at mailbox bytes 960..1023. A then reads the same device-side buffer. The actual descriptor source and destination pointers are recorded, not inferred from host arrays. Host diagnostic reads are made for verification, but these values are not copied back into the chip's feedback path.

## What separation means

The content payload contains 62 encoded content features plus a canonical marker at feature 0 and intentional alternate/swap marker at feature 63. It excludes the transient utility context, phase clocks, key-shadow carrier, knownness, return headers and control-status metadata elsewhere in the mailbox. The marker is deliberately part of the learning protocol; we do not claim absence of every management feature from the payload.

All content learning events use the same class teacher, label 0. Native competition chooses which content record holds a cue. Query-only phases leave complete content RAM unchanged. The four fixed routing/readout RAM banks are unchanged at endpoint audits. Both learner RAMs are read in full after every integrated forward.

## Primary and held results

| Run | Initialization and cue seed | Order seed | Retired physical record | New cue recalled | Protected recalls | Context checks |
|---|---:|---:|---:|---:|---:|---:|
| native06 | 27102741 | natural order | 4 | 1/1 | 7/7 | 32/32 |
| held_27102753_r2 | 27102752 | 27102753 | 3 | 1/1 | 7/7 | 32/32 |
| held_27102754_r2 | 27102752 | 27102754 | 4 | 1/1 | 7/7 | 32/32 |

Across these three working runs: new cue 3/3, protected recall 21/21, context invariance 96/96. These are two initialization-and-cue seeds, not three independent initializations. The local threshold of the selected record changes from 24 to 1 during retirement; the later cue reuses that record. Each protected record's complete RAM is byte-identical across replacement. An unfamiliar cue in the not-useful context leaves all content RAM unchanged.

The primary programme SHA256 is `0f04ba46c085096fa06c96fce6f252c3e740b54d19f351c90ed156d7a4374519`.

## Causal controls

| Matched physical arm | Manipulation | New cue recalled | Protected recalls |
|---|---|---:|---:|
| Learned manager | All mechanisms enabled | yes | 7/7 |
| Policy frozen | Initial policy trained; later reversal teaching disabled | no | 7/7 |
| Retirement inhibited | Native inhibitory input clamps retirement permission | no | 7/7 |
| Content frozen | Content updates disabled after initial fill | no | 7/7 |
| Normal threshold cap | Same retirement signal and connection swap, but no low-threshold capacity reset | no | 7/7 |

The retirement-inhibited arm is functional inhibition, not a literally disconnected physical wire. Immediately before new learning, the learned-manager versus normal-cap control differs at just one captured RAM word: record 4's local learning threshold, 1 versus 24. The weights, swap budget and all other RAM agree. In normal-cap, the marker swap can invalidate the old exact match, but does not admit the new cue. This distinguishes marking a record from actually making it reusable.

## CPU controls and claim boundary

Two bounded adaptive task-level CPU algorithms also solve the same memory task: an exact associative store with a learned context dictionary, and a nearest-Hamming store. They use the same encoded observations and explicit teachers, with eight content records and two context records, without neuron or learning-rule emulation. Separate Python and compiled C++ exact-rule models additionally match all native output and complete-RAM rows. This study establishes native execution of conditional replacement and isolation, not CPU inability, faster execution, energy savings or historical priority.

The experiment uses ten designed binary cue patterns, not real images or generalization to unseen views. It performs one replacement per working run. It has not established repeated turnover over many cycles, automatic utility teachers, autonomous victim search, physical fault recovery, world-model usefulness, or power-only operation. The host still writes external observations, role profiles and static global learning configurations, polls completion, and performs extensive verification reads.

## Accounting and preserved failures

Final accounting: 9 physical attempts; 3,950 reserved forwards; 3,887 observed forwards; 8 source qualifications. Observed forwards comprise one failed mailbox forward, 64 successful qualifier forwards, and 7 integrated cells x 546 = 3,822. The prospective physical-slot amendment accounts for the failed engineering attempt; no seed, endpoint or clock was relaxed after results. The card was released and there are no owned scientific jobs active.

Preserved attempts include the rejected V1 fan-in build; one-forward shared-ingress failure; missing Python import-path failure; a CPU reference using the old 128-input record stride instead of this bank's 256-input stride; an incorrect first dense-header auditor assumption; and NumPy-int JSON serialization during held preparation. The stride and serialization failures never reached hardware. The dense-header correction re-audited the same already recorded native trace; it did not rerun hardware. Exact failure/source receipts remain in this study. Media authoring also preserves the broken Homebrew Node dependency failure and an initial presentation that showed a stale B return during an A role; the final replay shows pending until a captured current-role return exists.

## Replay and reproduction

The film uses captured native input buffers, raw returned words and post-forward RAM. Cyan indicates a returned binary response; amber indicates captured connection edits. Stored record fingerprints and their exact-cue labels are decoded from measured RAM. Curves correspond to actual saved connections for the two eligible policy units and eight content neurons; geometry and camera motion are schematic. They are not electrical spike, voltage or physical-die measurements. The film selects 67 recorded events and presents them at 60fps; the presentation rate is not device throughput.

See `MEDIA_NOTES.md`, `MEDIA_QA.json` and the final review archive/clean-extraction receipt. The local archive contains owned source, raw data, reference outputs, prospective authority/admissions and preserved failures. It excludes the vendor SDK/runtime and unrelated inherited continuity. Nothing has been published to X or GitHub by this task.
