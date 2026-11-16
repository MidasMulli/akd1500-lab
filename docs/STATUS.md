# Current research status, November 16, 2026

This page separates physical results, numerical references and unbuilt ideas. All listed scientific campaigns are closed. Publishing their evidence does not authorize another hardware run.

## November results on physical AKD1500

- **Label-free learning from stored inputs:** one native learning layer produced the memories and full-panel scores behind 63.6–69.0% balanced atlas agreement, versus 18.3–23.1% with unchanged memory. The host encoded profiles and classified from chip scores using a fixed 60-label budget. Conventional software controls scored higher. One continuous capture repeats a prior run exactly.
- **Four-observation prediction:** on six synthetic streams, a fixed on-chip history and error circuit opened teaching only on the first error for each of two changed rules. Always-teach opened 48 gates. Gating and history controls establish which parts of the circuit mattered on those streams.
- **Connected learning:** all four learner banks changed connections on all 64 inputs in the all-learning session. Freeze controls preserved the frozen banks, and a change to the final fixed encoder changed fourth-bank output and the edited record as predicted. Every bank received the same external teaching label.
- **MechDog vision scoring:** the physical scorer replayed 309 saved frames and matched 618 host selections against the reference. A separate stationary live check cleared a known brown-box site when the box was removed. Acquisition of a relocated object and any use of the visual decision for motor control remain open; the saved finder missed five of 239 protected requests.

[Full November findings and downloads](NOVEMBER_2026.md).

## October 28 baselines

## Native managed image memory

Four12 × 8 silhouettes are stored as native pixel-value connections. A learned source checker regulates updates; partial cues retrieve the stored pixels. Primary and held-mask final exact recall are131/132 each. Held masks reuse the same objects, initialization and teaching sequence. The strong task-level CPU model passes316/316 queries in each cohort. Allocation/pixel addresses and presence teaching remain externally supplied.

[Study report](../studies/20261027-silhouette-managed-memory/REPORT.md) and [read-only explanation of the displayed abstention](../studies/20261027-silhouette-managed-memory/RODNEY_ABSTENTION_REVIEW_20261028.json). Its unique best car match scored10 on the quantized cue path; the preset cutoff required11. The source checker was not vetoing that read.

## First AKD/world-model retention study

The frozen model supplies h0/h5 states from a short actual observation. A frozen host PCA/bin encoder combines h0,h5,their difference and commanded program. No physics-setting, gain or scene-ID label enters the recall key. Later outcome labels teach associations; the host combines recall with model probabilities using the fixed75/25 rule.

Native all-outcome DEV K4 recall is64/64 returned queries across eight associations. A separate error-only DEV K2 test corrects16/16 changed-physics return forecasts versus0/16 for A-only frozen. Fresh selected native HELD K2 also reaches16/16. Adaptive CPU and native trained-on-A+B-then-frozen controls pass. Counts are cached repeated queries, not independent new scenes.

The v2/v3/v4 failed cohorts remain part of the ancestry; the sparse-magnitude cohort is the fourth fresh cohort in that sequence, not four independent successes. [Full report](../studies/20261028-wm-sparse-magnitude/REPORT.md) and [input-label audit](../studies/20261028-wm-sparse-magnitude/RODNEY_SETTING_LABEL_REVIEW_20261028.json).

## Causal Nano GPU stream with32 content records

Five physical sessions completed32,580 forwards and28,200 native device-buffer feedback windows. Every output, both learned RAMs, input/feedback descriptor and fixed-bank state passed its saved checks. The model's weights and encoder remained fixed; GPU fp32/TF32-off was qualified, while BF16 had an encoded-key mismatch and was rejected.

- DEV selected native:16/16 second-encounter forecasts, versus1/16 each actual native all-admit and matched-frozen control. This corrects15 model errors and retains16/16 earlier admitted corrections.
- HELD selected native:16/16 versus2/16 world-model/software references, correcting14 errors and retaining15/15 earlier admitted corrections. Only the selected arm was physically run on HELD.
- Strong adaptive CPU controls pass. DEV endpoint labels are15 neither and1 goal; this imbalance limits the challenge. Original-physics harm and earlier-memory retention are separate endpoints.
- K32 CPU diagnostic:29/29 old corrections retained, but only3/32 new forecasts correct, unchanged from the model. Full-memory replacement is unresolved.
- Unfiltered new-scene DEV:128 episodes,124 queries and4 observed-terminal skips. Every key was distinct, memory had0hits, and forecasts equaled the world model. **Usefulness gate failed.** No N HELD/native expansion or post-failure refit was admitted.

[Native DEV comparison](../studies/20261028-wm-stream-overnight/NATIVE_DEV_COMPARISON.json), [class balance](../studies/20261028-wm-stream-overnight/ENDPOINT_CLASS_BALANCE.json), and [stream report](../studies/20261028-wm-stream-overnight/REPORT.md). One zero-consumer-row startup failure and its bounded wait-only repair are preserved.

## Replacement proposal: software only

Two fixed seeds passed128 reference replacements while protecting all31 other record states after each update. A smaller three-arm circuit witness matched Python/C++ outputs and both full RAMs for3,168 forwards. SDK 2.19.3 accepted a virtual32-class configuration. Those are different checks, not128physical replacements. Both build-path failures are preserved.

The synthetic schedule/class label still supplies the destination. A native selector is not built or admitted. A five-record native allocation-pointer design is assessed only. Similarity-based useful recall is the first recommended next mechanism, followed by native replacement and a fresh integrated stream. [Proposal](../proposals/20261028-blind-replacement-source/README.md).

## Claim boundaries

No result here establishes speed/energy superiority, CPU-inexpressible computation, arbitrary unseen-physics recognition, autonomous host-silent operation, fresh closed-loop action selection or power-cycle persistence. Native connection learning, host encoding/teachers/clocks, forecast fusion, measured outputs and derived/schematic visuals are stated separately. The buffer's device-visible address is not by itself proof of its physical SRAM location.
