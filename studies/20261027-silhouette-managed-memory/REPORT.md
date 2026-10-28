# Native managed image memory — qualified bounded result

The physical network learned four 12×8 object silhouettes, rejected misleading image updates, accepted a legitimate image revision after source reliability reversed, and returned stored pixels from incomplete cues. All seven resident processors were FNP3, with internal weight memory. Two processors contain learning groups; five provide fixed comparison, gating and readout. This is a controlled memory component, not a general vision system or demonstrated CPU efficiency advantage.

## Measured endpoints

| Cell | Final exact image recall | Quarter-erased recall over all query phases | Random unfamiliar-input rejection |
|---|---:|---:|---:|
| primary | 131/132 | 284/288 | 160/160 |
| manager_frozen | 0/132 | 126/288 | 160/160 |
| open | 33/132 | 141/288 | 160/160 |
| content_frozen | 0/132 | 64/288 | 160/160 |
| cut | 0/132 | 64/288 | 160/160 |
| confirmation | 131/132 | 285/288 | 160/160 |

The support cell returned4/4 complete and64/64 quarter-erased images, rejecting16/16 random unfamiliar inputs. In each of the primary and held cells, all 28 complete-image reads, including the repeated native-buffer feedback reads, were exact. The held cohort changes only the erasure masks and random unfamiliar inputs; it uses the same four objects, initialization and teaching sequence. It is not a new-object or new-initialization study.

The primary and held adaptive cells each left the entire content-bank RAM unchanged on1,740 rejected teaching steps. Both accepted the two legitimate pixel revisions, producing two additional native connection swaps beyond initial teaching. The strong task-level adaptive CPU model passes 316/316 primary and316/316 held image queries. The independent compiled exact-rule CPU reproduces every native output and complete learning-bank RAM state in all seven cells. No CPU-exclusive, speed, energy or GPU-contention result is claimed.

## What is native and what the host supplies

The bootstrap teaches all four cue identities, then the first two pixel-content records. Later additions concern content for the other two already-taught cue identities. This is not unseen-object learning.

The chip receives a fixed binary encoding of the actual silhouette: three foreground rails and one background rail per pixel, plus source-context features and role flags. There is no fitted image encoder. The first learning processor holds image-key cells and a separate source-reliability group. Source quality is taught from externally supplied outcomes. A native4-bit score path and fixed pairwise circuit choose the image key without a CPU argmax. A native signed relay combines that key with the learned source response and gates the content features.

The content processor has 384 preassigned logical image/pixel cells, four learned presence cells and twelve protected helpers. Each pixel cell learns 31 identity features and one of two value features through the chip's native connection-update rule. A later query probes the value-one feature. Native readout uses the cell response and learned record presence to return both pixel-value rails. Fixed native stages gate the output on the learned presence marker and consistent pixel rails, then format 512 binary int32 words. They do not independently prove that every pixel write has completed. No host image table supplies the returned pixels.

The host supplies source observations, quality feedback, record/pixel teacher addresses, pixel values, learning clocks and complete-image presence teaching. **Allocation is preassigned, not learned. Presence/commit is announced by the host, not autonomously detected.** Updates and reads are staged; atomic concurrent revision, self-directed memory allocation, power-loss persistence and unbounded autonomous learning are not demonstrated.

## Processor and memory boundary

The installedSDK 2.19.3 AKD1500 target contains 32 neural-processor slots with overlapping supported modes. All 32 support CNP1; seven also support FNP3, and one also supports FNP2. These are capabilities of the same slots, not extra physical processor populations. Official BrainChip documentation defines FNP3 as internal-memory fully connected processing and FNP2 as external-memory processing. Our seven-stage mapping is entirely FNP3. Input/output scratch placement is a separate question and remains unqualified; internal learned weights do not imply zero external traffic.

Mapping: cue/source(1,3,0); comparison(2,1,0); gate(2,2,0); content(2,3,0); pixel readout(3,1,0); guard(3,2,0); raw formatter(3,3,0).

## Verification and failures

Seven owned physical sessions completed 22,554 observed forwards. Every raw binary output was read before SDK fetch and checked against the fetched output. Actual input bytes, complete learning RAM,64 direct binary content probes, source-probe rows and all five fixed RAM packets passed exact checks. Forty-eight output-address-to-input-address handoffs required no CPU input resend. These runs remain host paced and polled; they are not host-quiet or power-only autonomous operation. HRC parameters were not independently read back.

The initial software candidate failed because an auxiliary probe flag contaminated policy learning. The repaired binary cue interface also exposed cue coactivation and unfamiliar-input acceptance. A fixed native comparison/completeness path resolved those development issues before hardware. An 800-cell prototype-bank design mapped to external-memoryFNP2; our codec rejected that unqualified format, and that design was not physically run. R4 preserves 96-pixel objects but changes the storage code to one preassigned native cell per pixel. No failed candidate was promoted into a native result. The firstC++ build used a stale Python 3.10 include path; it was corrected to the installed Python 3.12 SDK. The initial media exporter had a local variable-shadowing error; the corrected export uses the same captured raw data.

## Film and provenance

The 45.42-second 4K60 and 1080p60 films replay actual native words and RAM changes. The stored-image panel is a separately labelled exact decode of captured connection masks; the returned-image panel is the actual native output. Sixty-four cell responses per event are directly probed. Active feature lines are circuit-derived from actual inputs and verified state. The source inset displays recorded source probes and variable-key connection changes; shared common-feature connections are omitted for legibility. Spatial layout and camera motion are schematic/editorial. No electrical spike waveform, physical floorplan, fabricated synapse, generated photo or device-latency timing is shown.

The film includes an actual abstention and unfamiliar input. Its final comparison uses the same query from the adaptive, open-gate and frozen-checker native captures. Full results, limitations and methods remain off film. Media QA verifies H264/yuv420p, 60 fps, 2,725 frames and complete decoding for both resolutions.

## Practical assessment and next target

This is useful progress from random balanced bit patterns to literal image-content memory under native learned admission and source reversal. It is still four designed silhouettes with externally addressed slots and designed feedback. The next milestone should remove the host-supplied completion stamp: a native coverage/readiness query followed by a native value query, using the chip's own returned metadata to control the second pass. First test interrupted initial writes; revision atomicity requires a separate generation/version design. Then connect the qualified controller to a sustained real observation/work stream and a finite native allocation/retention budget. Another decorative application or a CPU race on four symbols is not the target.

No new scientific cell is admitted. The card is released. The old overnight study, world-model arc and paused automations stay closed/paused. Tab5 remains delegated. This review package is local; no X or GitHub publication was performed.

Sources: https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.NP.Type and installedakida.virtual_devices.AKD1500. Numeric evidence: FINAL_NATIVE_RESULTS.json, per-cell native.AUDIT.json, raw native.* files, frozen source manifests and MEDIA_QA.json.
