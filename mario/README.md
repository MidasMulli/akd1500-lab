# First Mario consequence-learning screen

This screen tests a small visual/action association on physical AKD1500, then
executes its selected fixed actions in fresh emulator episodes. It does not
train a full Mario policy. The [protocol](PROTOCOL.md) was fixed before hardware
execution and includes the stop rules and exact task definition.

| Initialization | Learned | Frozen | Action-only |
|---|---:|---:|---:|
| 81701 | 7/8 | 6/8 | 5/8 |
| 81702 | 7/8 | 8/8 | 5/8 |
| 81703 | 7/8 | 6/8 | 5/8 |

The best constant macro scores 6/8. Two initializations beat both controls and
that constant, passing the predeclared small screen. The learned total is only
21/24 versus frozen 20/24, and learning harms the strongest frozen seed. These
are three initializations on one chip and the same eight nearby starts. This
is weak task evidence, not a statistical or robust generalization result.

One resident FNP3 bank contains 96 configured records, each selecting 32 of 512
binary inputs. The first two groups of 32 learn unsafe/safe associations; the
last group is protected for queries. Inputs come from an authored foreground
occupancy encoder over two RGB frames, placed in the candidate action's half
of the input vector. Game RAM is not an acting input. Outcome scoring uses the
emulator's reported death/position after the fixed horizon. The host supplies
two motor primitives and ranks native safe-versus-unsafe match margins.

All 752 native forwards—32 smoke plus nine 80-input cells—match the independent
integer/set reference, including every full RAM snapshot. Queries preserve
visible RAM; they can still advance the hidden selector. This does not prove
hidden-state restoration or power-loss durability. All 72 selected-action
trials match in fresh resets; this is execution of saved choices, not native
control on every video frame. CPU reference execution remains exact too.

## Reproduce in stages

Experience collection was run with Python3.13.12, gym-super-mario-bros9.1.0,
nes-py9.0.1, gymnasium1.4.0, numpy2.5.4 and Pillow13.0.0. Device fixture preparation
used the separately installed Akida2.19.3 compiler environment. The public
repository contains no runtime, generated vendor programs or game ROM.

```bash
# In an isolated emulator environment, collect actual rendered experience.
python mario/collect.py mario/.data01
# Read EXPERIENCE.json. Reject contradictory labels at identical encoded inputs.

# On the device host, in the compiler's environment:
python tools/new_study.py --name my-mario-screen --minutes 60 --sessions 12
export AKD_LAB_STUDY=studies/my-mario-screen
python mario/prepare.py mario/.data01 "$AKD_LAB_STUDY/fixtures"
# Inspect virtual checks, program identity and protected query thresholds first.
AKD_LAB_BENCH=learning_bench bash tools/build.sh
AKD_LAB_SNAPSHOTS=1 python tools/run_cell.py smoke32 \
  "$AKD_LAB_STUDY/fixtures/s81701_full" --samples 32 --cpu-repeats 1
python mario/verify.py "$AKD_LAB_STUDY" mario/.data01
```

Review the smoke before the full first seed. Stop if native/CPU/Python scores
or RAM differ, a query edits visible RAM, a resource/cleanup gate fails or the
first full seed does not beat the constant macro. Only then inspect frozen and
action-only controls and admit the remaining fixed seeds. Preserve every failed
attempt and do not change the held-out task or ranking after seeing results.
The normal external runtime/include variables from the root README are required.

For a completed study copied to this repo, with its original relative layout:

```bash
python mario/verify.py studies/my-mario-screen mario/.data01
python mario/replay.py studies/my-mario-screen/evidence/RECOUNT.json \
  mario/.data01 /path/to/new-fresh-replay
python mario/archive.py studies/my-mario-screen /path/to/new-fresh-replay
python tools/visualize_mario.py /path/to/new-fresh-replay --video
```

The current archive exporter intentionally checks the frozen 72-trial study
shape and writes dated result names. For a different task, prospectively version
those assumptions and destinations. `archive.materialize()` and the tests can
reconstruct every published score and snapshot without an emulator or hardware.

The next study should enlarge and seal the encounter distribution, vary timing
and geometry, retain strong frozen/constant controls and inspect representation
coverage. It must be a new protocol; this pilot should remain sealed.
