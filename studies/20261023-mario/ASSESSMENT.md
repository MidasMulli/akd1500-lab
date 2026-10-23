# First Mario screen closeout

Native execution and independent per-step verification passed in ten physical
sessions,752 forwards. The32-row smoke preceded the full first initialization;
the first full seed scored7/8 versus6/8 constant before controls and expansion.
Every native output and all captured RAM words match both reference paths.
All protected queries leave visible RAM unchanged. Original frozen source and
fixture hashes remained unchanged through closeout. The device was released,
and further admissions are closed.

The [results](../../results/2026-10-23-mario.json) show learned7/8 in all three
initializations, frozen6/8,8/8,6/8 and action-only5/8 each. The best constant
macro gets6/8. The predeclared two-of-three usefulness screen passes, but the
aggregate learned improvement over frozen is just21/24 versus20/24. One frozen
initialization performs perfectly and is harmed by training. This is a small,
mixed result, not strong evidence of reliable learned action selection.

The training set contains12 rendered-frame/action samples, repeated in four
fixed shuffled epochs. Six training prefixes and eight held-out prefixes share
the first encounter of the same level. Encoded train/test overlap is zero and
opposite-label aliases were absent; this does not establish independent scenes
or useful visual abstraction. No candidate encoder or ranking was tuned after
native readouts. No Mario physical attempt failed. Earlier scoping included an
already-dead prefix, retained outside the frozen physical study.

All72 selected-action trials matched in fresh emulator resets, with their exact
acting-input hashes checked. Four representative trajectories illustrate one
corrected action and one harmful update. The raw native scores and RAM are
[published](../../results/2026-10-23-mario.npz); the72 fresh game outcomes have
[separate receipts](../../results/2026-10-23-mario-fresh.json). Game outcomes are
emulated, while association updates and scores were measured on physical AKD1500.

The host supplies a binary pixel encoder, supervised experience schedule, score
ranking and two motor primitives. This is not full-game reinforcement learning,
learned perception, native frame-by-frame control or an unseen-level result.
The exact CPU model reproduces every measured update; electrical advantage and
restart/power-loss persistence remain untested.

The next series should use a new, prospectively sealed encounter distribution
and stronger representation coverage. Retain all frozen/constant/observation
ablations and judge fresh outcomes before expanding. The current pilot remains
sealed, including its harmful update and stronger frozen seed.
