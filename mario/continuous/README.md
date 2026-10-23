# Continuous native Mario decision loop

The chip queries two actions, chooses one through the unchanged host margin
ranking, executes 24 game frames, receives the actual short outcome and chooses
again. One physical native program stays resident across four environment
episodes. Neural state is not reloaded between those episodes. The control loop
ends on actual death, flag, environment termination or 1,200 controlled frames.
This is continuous action selection rather than telemetry beside a committed
90-frame action. Supplied prefixes and between-episode resets remain explicit.

**The first pair failed the development gate.** Learned advanced 1,418 pixels
versus frozen 1,554; both died in every episode. No additional seed or prospective
confirmation start was admitted. Read the [protocol](PROTOCOL.md),
[assessment](../../studies/20261023-mario-continuous/ASSESSMENT.md) and
[machine-readable results](../../results/2026-10-23-mario-continuous.json).

## Mechanism and limits

The unchanged authored projection+clock encoder maps current RGB, RGB eight
frames earlier, observed emulator frame count and candidate action into 1,024
binary inputs. One FNP3 bank on NPU `(1, 3, 0)` holds 192 records, each with 32
selected connections. The unsafe and safe groups have 64 records each; a third
protected group serves query labels and is excluded from ranking. Acting inputs
never contain game RAM, position, death or future outcomes.

The host queries A0 and A1 with protected labels, compares native
`max(safe) − max(unsafe)`, and breaks ties toward A0. A0 holds right+jump for
eight frames then right for 16; A1 holds right for 24. Actual survival and
displacement after execution generate the teaching label: safe iff alive and
advance ≥12 pixels. The older decision input is taught, not the post-action
observation. Frozen sees identical training inputs with protected labels and
suppresses all online updates. Arms start with byte-identical RAM and programs;
their subsequent paths differ when their choices differ.

There are 96 training contexts, both actions evaluated by fresh prefix replay,
four class-balanced epochs and 1,504 presentations. Only four of the 192 raw
action outcomes were unsafe. Repeating them for class balance does not create
new hazardous situations. This is an important coverage limitation, not a
proven explanation of every failure. No encoder or outcome-driven parameter
change was made after collection. The new action duration and labels make
direct accuracy comparisons with the older 90-frame task invalid.

## Read or reproduce

Hardware-free public checks need NumPy and Pillow:

```bash
python -m pip install numpy pillow
python -m unittest discover -s tests -v
python mario/continuous/audit_public.py
python tools/visualize_mario_continuous.py --video
```

The archive includes all native inputs, labels, initial/final RAM, native/C++
scores and every per-step RAM snapshot. `model.py` provides the stateful
integer/set controller reference; its outputs and state are tested against a
previously published 1,376-forward physical stream. Native execution still
checks the separate C++ reference inline, and `tools/verify.py` independently
recounts the complete raw stream. The public tests reject substituted actions,
reassigned query frames and visual sampling of a later episode after an arm dies.

Actual gameplay recording and fresh-frame audit use the emulator environment
documented in `mario/sealed/README.md`. Bring your own authorized emulator/ROM
and external Akida compiler/runtime; none is redistributed here. The physical
stream binary is the previously validated `mario_stream`; no installed driver,
runtime or firmware was modified. `serve.py` is its bounded owner adapter for
the new study, with the common device lock, source hashes, caps and cleanup.
Completed study states intentionally refuse another launch.

For a new prospectively specified study, freeze the protocol and producers
before collecting outcomes. Read the resulting data adequacy and reference
checks before compilation, then inspect virtual outputs, a mixed physical
smoke, the first matched pair and its independent recount before seed/test
expansion. This completed pilot's producer uses fixed study shapes and dated
destinations; version those deliberately for another protocol.

```bash
python mario/continuous/collect.py mario/.data-new-continuous
# In the external compiler environment:
python mario/continuous/prepare.py mario/.data-new-continuous \
  studies/my-new-study/fixtures-loop01
# On the emulator host, after separate source/admission/smoke gates:
python mario/continuous/loop.py studies/my-new-study \
  studies/my-new-study/fixtures-loop01/s84101_full /path/to/new-capture \
  --native-name development_full01 --wall
```

The validated SSH alias/workspace/interpreter are explicit in `loop.py` and
must be configured and re-frozen for another installation. CPU reference and
constant-policy runs omit `--native-name`; their reports say `physical:false`.
`--constant` is a CPU diagnostic override and was never used on the physical
pair; both local and public auditors enforce that distinction. `audit.py`
replays every rendered frame, motor button, feature vector, decision and
outcome in fresh emulator episodes. `archive.py` publishes only audited runs.

## The paired film

The [4K60 film](../../visual/mario-continuous-4k60.mp4) contains four matched
development starts. Runs were recorded sequentially on one chip and aligned
at each episode start. Individual source pauses remain. There is no reset
within an episode, time-compressed gameplay or invented electrical firing.
A darkened arm means it died while the other continues. That arm's neural
display is capped at its own episode end, so responses from its later episode
cannot appear on ended gameplay. Gold marks a recorded RAM edit; a recent-edit
outline persists for 0.3 seconds with a displayed age. It is not a spike-duration
measurement. Blue/purple link slots are actual selected image/time addresses.
The layout is logical, not physical die geometry.

The public prefix-image bundle is **deterministic fresh prefix replay**, checked
against the first captured frame. It supplies the eight-frame image history
before the first controlled query; it does not claim live prefix timestamps.
Controlled gameplay images and native query timestamps come from the capture.

No reliable learning advantage, full-level completion, learned perception,
autonomous search, CPU/energy advantage, full hidden-state restore or power-loss
persistence is established. The immediate next scientific priority is broader
hazard/consequence coverage and representation tests before a new sealed run.
