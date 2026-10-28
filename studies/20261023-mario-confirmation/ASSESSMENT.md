# Prospective Mario confirmation: learning-effect gate failed

The [new protocol](PROTOCOL.md) was fixed before these test outcomes were
collected. It asked whether the unchanged development-selected pipeline shows
a consistent learning effect versus matched frozen controls. The earlier
usefulness gates remain failed and their original 128 test contexts unopened.
This is a separate endpoint and bank, not a relabeling of those failures.

| Initialization | Learned | Frozen | Best constant |
|---|---:|---:|---:|
| 82911 | 110/128 | 106/128 | 111/128 |
| 82912 | 109/128 | 106/128 | 111/128 |
| 82913 | 111/128 | 112/128 | 111/128 |
| 82914 | 111/128 | 103/128 | 111/128 |
| 82915 | 110/128 | 107/128 | 111/128 |
| Total | **551/640** | **534/640** | **555/640** |

The primary gate required at least four pair wins, an aggregate gain of at
least 5 percentage points and a positive lower conditional block-bootstrap
95% bound, after the task adequacy gate. Four pairs favored learning, but
**the gain was only 2.65625 points**. Among the 25 conditional contexts where
exactly one action succeeds, the mean gain was 13.6 points. Its prespecified
level/wait block interval was **−2.86 to +25.6 points**; the additional
hierarchical level-then-wait interval was −13.75 to +31.25 points. Both include
zero. These intervals describe the conditional difference, not the all-valid
2.66-point difference. **Learning-effect gate: FAIL.**

The finite bank contains 128 valid contexts, including 25 conditional contexts
with successful action identities split 9/16. Training/test encoded overlap is
zero. Both actions succeed in 95 contexts and neither succeeds in eight. Thus
most all-valid opportunities do not distinguish the controller's choice. Five
initializations share the same bank and chip: this is not 640
independent environments. Three of the four level families contribute almost
entirely action-insensitive cases; known level/timing structure and the small
conditional set limit the strength of inference. There is no unseen-level
generalization claim.

All 11 physical sessions (mixed smoke plus ten full/frozen cells) passed exact
native/C++ and independent Python per-forward score and RAM checks: 13,792
forwards. Every cell has a 1,120-input training prefix followed by 256 protected
test queries, no held-out prequeries, and no test teaching. Frozen arms see
the same inputs with protected labels throughout and keep all visible RAM
unchanged. Query RAM is unchanged in every arm. Hidden selector restoration
and power-loss persistence remain unproven.

Compiler-only inspection and the 32-input smoke passed before the first pair.
The first pair scored 110 versus 106, and the prospective task adequacy counts
passed; remaining fixed pairs were admitted sequentially after review. No
representation, ranking, initial-weight, class-balancing or epoch change followed
test inspection. Initial weights were checked against seed-only generation.
All 1,280 native-selected action outcomes matched fresh emulator resets, with
RGB/time acting inputs reconstructed too. These are ten policies on 128
contexts, not 1,280 independent worlds or frame-by-frame replanning.

CPU-only action-only forecasts scored 104/128 in every seed. Shuffled-feedback
forecasts scored 108, 107, 110, 110, 107 (542/640). These are model controls,
separate from the physical learned/frozen arms. The best constant exceeds the
learned aggregate, and full physical utility controls were not run. **A useful
Mario policy is not qualified.** A positive mechanism does not overcome this
negative task result.

Two mechanical producer corrections are preserved: a duplicate-key construction
was fixed before compiler execution; an attempted build then failed on a Python
`prepare` module-name collision before data/compilation/device access. Invoking
the validated producer as a standalone subprocess repaired it. Original source
versions, failure log and lock versions remain saved, with before/after hashes
in [REPAIRS.json](evidence/REPAIRS.json). The actual dataset receipt points to
the original pre-outcome lock; later lock versions do not replace that history.
These repairs changed producer plumbing after collection, not the scientific
pipeline. Compiler intermediate files containing prequeries were retained as
unadmitted and were never used on hardware.

The [live film](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/mario-live-responses-4k60.mp4) shows a separate
development capture with real sampled chip responses and 45 native connection
replacements. Its gameplay is not this held-out evaluation. The film's table
is this confirmation and explicitly says the gate failed. Chip activity is
measured match output and RAM, not an electrical spike trace.

The full extension closed physical admissions after 21 sessions and 25,406
verified forwards, under the inherited 24-session/150,000-forward cap and
deadline. The device was released. No further representation was admitted.
Next work needs a new development protocol for local motion/hazard features,
shorter revisable action commitments and clock-only/constant controls before
another sealed evaluation. More seeds of this pipeline would not address its
observed failure to improve on the constant action.

[Machine-readable results](../../results/2026-10-23-mario-confirmation.json),
[all physical inputs/labels/scores/RAM](../../results/2026-10-23-mario-confirmation.npz),
[fresh selected outcomes](../../results/2026-10-23-mario-confirmation-fresh.json),
[reproduction and claim boundaries](../../mario/sealed/README.md).
