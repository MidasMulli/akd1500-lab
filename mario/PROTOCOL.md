# First Mario consequence-learning screen

This is a bounded visual-action consequence test in actual Super Mario Bros1-1,
not full-game reinforcement learning. No game RAM is an acting input. The host
supplies a fixed image encoder, training schedule, two motor primitives and
action ranking. Learning of selected binary connections takes place on AKD1500.

Use gym-super-mario-bros9.1.0 / nes-py9.0.1, seed81701. Each sample resets the
emulator and walks right for a fixed prefix. Training prefix lengths are
106,110,114,126,130,134 frames. Held-out lengths are108,112,116,124,128,132,136,138.
These are held-out starts around the same first encounter, not unseen levels.
Early scouting used different/coarser starts to establish a nontrivial task:
an always-jump-immediately macro and an always-delay macro do not both succeed
everywhere. Scout attempts, including an invalid already-dead prefix, are retained.

Two supplied 90-frame primitives: (0) hold right+jump for20 frames then right;
(1) hold right16 frames, right+jump20, then right. Success is alive with x>=350
at the fixed horizon. Game info is used only for outcome scoring, not the visual
encoder or action ranking. Death/no progress is not a hand-written policy.

Encoder: foreground occupancy in a fixed16x8 grid over pixels y96..223, using
the current frame and frame8 earlier; concatenate256 bits. Foreground differs
from the reset sky reference by>25 in any RGB channel and occupies>=24 pixels
per16x16 tile. Put the256 visual bits in the chosen action's half of512 inputs.
This is an authored pixel representation, not learned perception. Before device
admission, reject opposite labels at identical encoded visual/action inputs.
Report train/test encoding overlap rather than silently treating it as novelty.

One resident FNP3 bank:96 neurons,32 connections each, three classes. Classes0/1
learn unsafe/safe outcomes; class2 is a protected query class whose learning
threshold32 cannot be exceeded by32 selected links. Query requests can advance
the hidden selector, but must leave every captured RAM word unchanged. Use the
existing exact native/CPU/Python selector audits, not restored hidden state.

Three fixed initialization seeds81701/2/3, optimizer inherited from the proven
binary-association harness (.5 initial,.25 minimum,.125 decay,.25 competition).
Stream:16 protected pre-queries, four fixed shuffled epochs of12 recorded
experience/action outcomes,16 protected post-queries. Rank the two candidate
actions by max(safe match)-max(unsafe match), tie action0. Compare all-query frozen
weights and an action-only constant-cue ablation, plus best constant macro.
Native-selected actions must then be replayed in fresh emulator episodes and
their actual outcomes checked. Training uses recorded experience, not a claim
of an autonomous online policy discovering its own motor primitives.

Stages: emulator repeatability/input-source/alias checks; virtual mapping and
initial score proof; one32-sample physical mixed query/learning smoke; one full
80-sample seed with independent per-step RAM/output recount; inspect usefulness
before expanding other seeds/controls. Stop on source drift, alias gate, native
mismatch, query RAM change, resource ownership or failed cleanup. Preserve all
failures. Do not tune to held-out readouts or change the task after physical runs.

Useful visual-consequence gate: exact native mechanics, improved held-out action
success over both frozen and best constant macro by at least one start in at
least two of three seeds, and more success than action-only ablation. The first
seed failing to beat the constant macro stops expansion pending a new prospective
study. A pass establishes this small skill only; no raw-pixel end-to-end learning,
full Mario competence, new-level transfer or CPU/energy advantage follows.
