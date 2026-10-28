# AKD × world model: staged result and closeout

**The combined quality gate failed; no AKD hardware session was opened.** We completed three prospective CPU-stage cohorts with Omarchy, preserved all rejected encodings/failures, and did not turn a passing small cell into a hardware success claim.

The intended capability is an experience memory beside a frozen world model: observe the first five action ticks, forecast the remaining35, query stored outcomes, then admit the real outcome only after the episode finishes. The model weights stay frozen. The memory/controller in this test was an exact-rule software circuit, not a live chip.

| Cohort | Fresh HELD two contexts: returned A / B | Fresh HELD four contexts: returned A / B | Result |
|---|---|---|---|
| v2 | 100.00% / 100.00% | 100.00% / 100.00% | WM already100%; no changed outcomes in primary HELD contexts |
| v3 | 87.50% / 87.50% | 93.75% / 93.75% | One alias in the ten-effective-bit representation |
| v4 | 100.00% / 100.00% | 93.75% / 93.75% | Fourteen-bit repair still aliases one K4 A/B pair |

The adaptive full-precision CPU control retained100% at both primary capacities in every cohort. Its purpose was to show whether the observed latent contained the missing information; it did. This is a representation failure before a hardware comparison, not evidence that the AKD is intrinsically unable to learn the task.

The final failure is concrete: the same scene/program reaches the goal under gain1.0 and times out under gain0.6. h0 is identical; h5 differs by L2=0.07536. A PCA component changes from−0.00355 to−0.06379, but both become the same sign bit. All14 signs match. The memory therefore receives the same key for two different outcomes and truthfully revises that one record, causing errors when the old regime returns. K4 held recall was93.75% in both returned regimes.

Two source-level improvements are retained: distinct keys are duplicated so a one-feature change cannot accidentally satisfy the native update threshold; and removing constant/program overhead increased effective information from10 to14 bits within the same32 connections. Both passed source mechanism checks. The latter did not solve all unseen cases.

K8 diagnostics exceeded eight-record capacity (and sometimes aliased) and failed. They are included, not discarded. These are repeated exact cached context/gain episodes, not independent natural video samples. V3/v4 were explicitly simulator-qualified changed-outcome challenges, selected without consulting WM predictions; they do not estimate deployment performance.

The error-only arm does not store a correct baseline answer. A correct fallback prediction is therefore not counted as native memory retention. Once both regimes are already stored, the fully AB-trained frozen control also passes; continued learning has no demonstrated benefit on those identical recurrences.

Independent scalar recomposition and saved-RAM audits checked 19,800 software query records. No physical forwards, GPU contention, energy, live control or planner improvement was measured in this work.

The next target is magnitude-preserving sparse input, not another task or more sign bits. Thirty-two learned input addresses can select among a larger set of feature bins; they do not require compressing the whole state to32 scalar signs. Wider geometry and routing must be qualified first. See NEXT_EXECUTABLE.md.

The proposed SDK builder, actor, observer readouts and native protocol remain UNRUN/NOT ADMITTED. The review package is local. No X draft, public repository, deployment, Tab5 serial owner or console service was changed.
