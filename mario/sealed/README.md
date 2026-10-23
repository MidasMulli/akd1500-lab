# Sealed Mario consequence learning and live native telemetry

This extension tested a larger visual/action bank on the physical AKD1500.
The final [prospective confirmation](CONFIRMATION.md) **failed its learning-effect
gate**: learned 551/640, frozen 534/640, constant 555/640. Four of five paired
initializations favored learning; the aggregate gain was below the frozen
threshold and the conditional resampling interval included zero. See the
[assessment](../../studies/20261023-mario-confirmation/ASSESSMENT.md) and
[raw-derived report](../../results/2026-10-23-mario-confirmation.json).

## What runs where

The Mac emulator produces current RGB, RGB from eight frames earlier, elapsed
emulator frames and candidate action. A fixed encoder creates 1,024 binary
inputs: one action-local half containing 256 image and 256 clock features.
The clock is observed frame count, not game RAM, wall time or a future label.
One physical FNP3 bank on NPU `(1, 3, 0)` contains 192 logical neuron records
with 32 selected connections each. Groups of 64 learn unsafe/safe associations;
the third group is protected for query labels. The host ranks native
`max(safe) − max(unsafe)` and commits one of two supplied 90-frame actions.
Evaluation alone reads the emulator's death/position information.

Training uses 80 saved encounter contexts, two action outcomes each, and four
class-balanced epochs: 1,120 presentations. Confirmation trains first, then
queries 128 distinct timings with both candidate actions. Frozen arms receive
the same inputs with protected query labels throughout. There are no test
prequeries or test teaching. Query RAM is unchanged, but hidden selector state
can advance; no state-restore claim follows from a RAM comparison.

## Evidence and stage history

1. [Original development](../../studies/20261023-mario-sealed/ASSESSMENT.md):
   two RGB-only encodings failed calibration and each aliased a contradictory
   observation. No physical sessions; original 128 test outcomes unopened.
2. [Revision 1](../../studies/20261023-mario-sealed-r1/ASSESSMENT.md):
   observed clock plus balanced experience removed that alias, but usefulness
   calibration failed again. Six native diagnostic cells and live captures
   verified the mechanism; those are development results.
3. [Separate confirmation](../../studies/20261023-mario-confirmation/ASSESSMENT.md):
   the selected pipeline and five seeds were fixed before new test outcomes.
   Compiler checks, mixed smoke, first pair, remaining fixed pairs and fresh
   emulator replays were checked in that order. No neural tuning followed test
   inspection. The final gate failed.

The original test bank and the new confirmation bank have different timing IDs.
The former remains unopened. Neither is an unseen-level benchmark. All five
seeds share 128 contexts on one chip. The 1,280 fresh replays execute ten saved
policies on that same bank; they are not 1,280 independent worlds.

Public archives contain native inputs, labels, initial RAM, native/C++ outputs,
per-step RAM and final RAM. The Python integer/set reconstruction independently
recounts every forward, including selector updates. Mutual agreement between
native and C++ outputs alone is insufficient; forged matching outputs are
rejected by the public tests. Native action-only and shuffled-feedback controls
were not run; their explicitly labeled CPU-model forecasts remain separate.

## Inspect without hardware or an emulator

```bash
python -m pip install numpy pillow
python -m unittest discover -s tests -v
python tools/visualize_mario_live.py --video
```

The movie renderer uses FFmpeg's `h264_videotoolbox` encoder on macOS. Choose an
available H.264 encoder to render on another host. It rebuilds the film from
[recorded game frames](../../visual/mario-live-game.npz),
[timestamped events](../../visual/mario-live-data.json) and measured RAM.
The video is 3840×2160 at 60 fps. It holds the last captured frame through real
pauses; it does not invent intervening chip responses. Mac receive times bound
what can be shown. No cross-host clock subtraction is used.

The capture shows selected development cases, including failures, with native
responses sampled while the game runs. Four actual outcomes trigger 45
connection replacements. Shadow queries target 10 Hz and achieve 8.4 pairs/s;
they do not revise the committed macro. White outlines denote a rise between
sampled match values; gold marks a recorded RAM edit. Logical placement is
schematic. Electrical spikes, transistor geometry and individual cycle timing
are not observed.

## Reproduce a new study in stages

Do not reopen the completed study or reuse its authority. Create a new namespace
and freeze its task IDs, scientific pipeline, source hashes, limits and gates
before collecting its held-out outcomes. The checked environment was Python
3.13.12, gym-super-mario-bros 9.1.0, nes-py 9.0.1, gymnasium 1.4.0, numpy 2.5.4
and Pillow 13.0.0. The licensed compiler environment was Akida 2.19.3. Bring
your own authorized emulator/ROM/runtime; none is redistributed here.

```bash
# First collect training/development only; this manifest includes unopened test IDs.
python mario/sealed/collect.py /path/to/frozen-manifest.json \
  mario/.data-new-dev --splits train dev --clock

# In the external compiler environment, prepare virtual programs and inspect them.
python mario/sealed/prepare.py mario/.data-new-dev mario/.data-new-dev \
  studies/my-new-study/fixtures-cal --phase calibration
python mario/sealed/calibrate.py studies/my-new-study/fixtures-cal \
  studies/my-new-study/cpu_runs/calibration
```

Read the resulting gate before admitting a physical smoke. The exact CLI and
fixed study shapes are in the versioned source, not a general training API.
`tools/new_study.py`, `tools/run_cell.py` and the normal runtime/include build
variables described in the root README supply bounded execution. Compile
`learning_bench` for fixtures and `mario_stream` for interactive telemetry:

```bash
AKD_LAB_BENCH=learning_bench bash tools/build.sh
AKD_LAB_BENCH=mario_stream bash tools/build.sh
```

`confirm_collect.py` requires an existing matching design manifest and a pipeline
lock whose source hashes still match. `confirm_prepare.py` invokes the validated
compiler producer and removes every pre-training test query. It preserves the
unadmitted intermediate. `verify_confirmation.py` checks actual train/test
boundaries, protected query RAM, per-step outputs and paired task outcomes.
`confirm_replay.py` executes native-selected choices in fresh emulator resets.
The published archive exporter intentionally enforces this completed study's
shape; version its assumptions for a different study.

`live.py` is the validated Mac-to-Nano capture harness. Its SSH alias, remote
workspace and compiler interpreter currently match the recorded installation;
configure those explicitly for another system and freeze the new version.
`serve.py` requires a prospectively admitted calibration/live study, checks source
hashes, uses the common device lock, refuses another owner, applies forward/time
caps, preserves failures and releases only its own native child. Closed study
states intentionally refuse another launch.

There is no full-game RL, learned visual perception, useful policy qualification,
CPU/energy advantage, complete hidden-state restore or power-loss persistence
claim. The next scientific work needs a new representation/control development
gate, not more seed search on this sealed bank.
