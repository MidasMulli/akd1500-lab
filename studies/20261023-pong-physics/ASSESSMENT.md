# Changing-physics Pong — first development screen

**Stop the scientific expansion.** Native learning improves the matched
comparison after the physics changes, but its absolute prediction quality is
not useful enough to admit a paddle controller.

| Endpoint | Continue learning | Trained, then frozen |
|---|---:|---:|
| Original physics, separate development trajectories | 63/96 | 63/96 |
| Changed court before further feedback | 4/48 | 4/48 |
| Changed court after further feedback | 12/48 | 4/48 |
| Mean landing-position error after feedback, court height = 1 | 0.294 | 0.412 |
| Unchanged court, after further feedback on both courts | 39/48 | 32/48 |
| Connection replacements after training in the matched bank | 5,996 | 0 |

The 16.7-point matched gain and 28.6% landing-error reduction pass those
individual screen requirements. The 25% changed-court accuracy fails the
prospectively fixed 65% requirement. Its label distribution is 14, 19, 11 and
4; the evaluation-only majority constant reaches 19/48. Four categories do not
imply a balanced chance baseline.

Both arms reached byte-identical trained RAM and exact prefix scores after
the same 512 native labeled forwards. Their subsequent inputs and query order
match. Frozen permits no further visible RAM changes. No hidden-state restore
is assumed. The original-physics probe passes its 65% gate by one example.

On the changed-court teaching examples, scores *before* each update improve
from 9/64 in the first pass to 39/64 in the sixth. This is a post hoc diagnosis
of fitting, not a new experiment or confirmation result. The gap to 12/48 on
the separate development trajectories makes generalization a concrete next
target. It does not identify one unique hardware or algorithmic cause.

The unchanged court gets additional correct teaching during adaptation.
Therefore 39/48 is retention with rehearsal, not proof that original memories
survive with no refresh. Development outcomes were also inspected to reject a
coarse encoding that aliased different landings before any physical run.
These results are development evidence. The separate confirmation bank and
remaining initialization seeds stay unopened.

## Visual evidence and execution

Five physical sessions include a 32-forward smoke check, two 1,520-forward
matched arms and two 534-forward live captures. All **4,140** forwards, native
scores, connection selections and full per-step RAM snapshots agree with the
inline C++ reference and independent Python integer/set reconstruction.

The live captures have 16 wall-paced serves per arm and six post-training
feedback opportunities. Continued learning makes 174 actual connection
replacements; the frozen arm makes zero. Queries arrive before launch, labels
are sent after landing, and every recorded simulator position reconstructs
independently. The film is a 4K/60 fps replay with source rates near 57 fps;
query pauses are preserved. The two physical runs occurred sequentially.

Paddles remain at the centre: the native predictions are shadow observations.
This is not learned game control, successful full games, pixel learning,
native reinforcement learning, electrical spike timing or an efficiency
advantage. The chip receives a supplied motion representation and learns
binary associations natively. The 160 displayed records occupy one mapped
learning bank, not all 32 NPUs.

The failed compiler-only 320-neuron attempt selected a memory format the
decoder cannot verify; this is a tooling limitation. Its receipt and source
are preserved locally. No physical attempt is discarded. Admissions are
closed after the failed gate and diagnostic capture; all owned programs are
unprogrammed and the shared device lock is released.

## Next scientific step

Before more gameplay, separate three questions: how well the current circuit
can fit the changed teaching bank, how well it predicts new motion examples,
and whether old court-B knowledge persists when B receives no further
teaching. Compare a small, prospectively fixed motion-code/prototype capacity
grid on fresh development trajectories, with constant and simple motion-model
baselines. Preserve the same shared-trained frozen control. Only a new passed
prediction gate should admit useful paddle-control testing.
