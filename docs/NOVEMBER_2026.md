# November 2026: results from the physical AKD1500

[Watch the continuous live take](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261116/AKD1500_atlas_live_1080p60.mp4) · [View the release](https://github.com/MidasMulli/akd1500-lab/releases/tag/research-20261116) · [Verify the downloads](REPRODUCING.md#november-2026-downloads)

## Native label-free memory: three repetitions and one live repeat

In each repetition, one learning layer received 8,192 saved, encoded profiles from one mouse, without class labels for the learning step. The host supplied 60 labelled examples to name memory records, then used the chip's scores to classify the same 9,511-cell external panel from another mouse. The chip did not receive cell positions; the host uses them to draw the map. The 73-second 1080p60 video preserves the full captured sequence, including learning, the switch to frozen memory, live scoring and release.

| Repetition | Label-free memory | Untrained memory | Supervised native control |
|---|---:|---:|---:|
| 1 (shown live) | 69.0% | 18.3% | 61.1% |
| 2 | 67.9% | 23.1% | 59.5% |
| 3 | 63.6% | 22.6% | 63.3% |

Agreement is balanced across classes and sections. The continuous live repeat matched the earlier run across all 17,763 calls and 18 readbacks of memory. All 9,511 decoded panel predictions matched the reference; this is an exact repeat on the same panel, not perfect atlas classification or another generalization test. Conventional exemplar and k-means software controls scored higher. The label curve uses physical scores only at the marked four-label-per-type AKD points; its other points are software results.

Data and map credit: [Xiaowei Zhuang and colleagues, Allen Brain Cell Atlas mouse MERFISH data](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang_dataset.html), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Maps show model predictions drawn for this experiment.

## Prediction-error gated teaching (HAL3)

On each of six synthetic binary streams, a fixed circuit kept the last four observations, and another fixed circuit opened teaching when the chip's prediction disagreed with the supplied next bit. After a rule change, this opened **2 of 48** teaching gates, once for each changed history. Always-teach opened 48. Both arms got every later revisit of the two changed histories correct and preserved the registered unchanged answers. Cutting the teaching gate or freezing the learner left the old responses; replacing the history with a fixed value introduced errors. The initial rule and history, observation streams and circuit design were supplied externally. Gate counts do not measure connection writes, latency or energy.

## Four connected learning banks (FOUR-64)

Four connected 64-record banks on one chip changed connections on every input of a 64-input synthetic stream. The complete matrix comprised eight sessions, 390 forwards and 1,288 learner memory captures. Each bank got the same externally supplied class label and updated using the chip's local rule. In the freeze controls, frozen banks stayed byte-for-byte unchanged. Changing the last fixed encoder altered the fourth bank's score vector and edit as predicted. This does not establish backpropagation, independently learned encoders or whole-chip capacity.

## MechDog: scoring on AKD, region selection on the host

The chip did the patch-template comparisons for **309 saved frames**. All **618 host region decisions** matched the reference. The fixed scheduler used 888 patch-batch pairs versus 3,708 for the whole-frame reference schedule (76% fewer pairs, not a hardware speed or energy measurement). In a fixed 89-view shifted subset of the walks, selection with recorded AKD scores localized the target in **85/89** views versus **80/89** for host-only ranking. The wider gate still missed five of 239 protected old requests; the all-walk result was 133/137 pairs, with zero false calls on the separate 24-request absence panel.

In a separate stationary live check, AKD scored patches at a known site in six camera frames: the brown box was present in five, and the decision cleared in the final frame when the site was empty. The capture did not include a return or establish whole-frame rediscovery. Motors were off. The host still encoded inputs, proposed regions and selected the displayed result. Neither run demonstrates motion control, a learned visual feature extractor on the chip or generalization to new rooms and objects.

The release includes a continuous video, the atlas figures, and small evidence tables with a software-only verification script. The tables are an inspectable projection of the checked results. They do not contain chip programs, private recordings, or the raw datasets, and they do not provide a turnkey hardware rerun.
