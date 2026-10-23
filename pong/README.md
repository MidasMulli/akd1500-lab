# Changing-physics Pong

This experiment asks whether a native AKD1500 predictor can recover after an
unannounced physics change. It deliberately starts with identical recorded ball
trajectories, before testing a learned paddle controller.

**The first development gate failed.** Continued native learning predicted the
correct changed-physics landing band on 12/48 trajectories, versus 4/48 for the
trained-then-frozen control. The evaluation-only best constant reached 19/48.
Additional seeds, the confirmation bank and learned paddle control were not
admitted. The video is a measured shadow diagnostic with fixed paddles.

## What the chip receives and learns

The host supplies observed ball height, vertical velocity and visible court
identity as a fixed 1,024-bit population code with 64 active inputs. Future
acceleration, landing position and target class are absent from prediction
inputs. A native binary learning layer has 160 neuron records, 64 selected
connections per record and four landing-band groups. A fifth protected group
allows queries without modifying visible RAM. A query still advances internal
selector history, so query order is matched as well.

The ball crosses the court in 1.2 simulated seconds. Court A's vertical
acceleration changes from +0.8 to −0.8; court B stays at +0.3. Top and bottom
walls reflect the ball. The chip predicts one of four landing bands, not a whole
trajectory. The host supplies the simulator, observation encoding, class-maximum
decoder and lowest-index tie rule. This is supervised association on a small
motion representation, not pixel learning or native reinforcement learning.

Both physical arms replay the **same 512 labeled training presentations** and
the same query prefix. Complete trained RAM and prefix scores match exactly.
One arm then receives the recorded landing labels; the other receives only
protected labels. Initial visible RAM is never restored as a substitute for
replaying the native learning history. Both see exactly the same inputs and
query order. Each forward is checked against the C++ reference, and an
independent Python integer/set reconstruction checks every score and full RAM
snapshot afterward.

The matched bank contains 96 original-physics development probes, 48 changed
probes before feedback, 768 adaptation presentations, 48 changed probes after
feedback and 48 unchanged-court probes. The changed accuracy gate was 65%,
alongside gain, landing-error and retention requirements. Development outcomes
were inspected for encoder aliasing before hardware execution; they are not
sealed confirmation evidence. Court B also gets further correct training, so
its final score is a retention-with-rehearsal endpoint.

## Recorded visual

`visual/pong-physics-4k60.mp4` is a 4K/60 fps export of two sequential native
captures aligned at their starts. Each shows 16 wall-paced serves. The source
simulation samples achieved about 57 fps, including native-query and feedback
pauses. The film reconstructs those recorded positions without speeding up
time. Native prediction packets arrive before launch; teaching labels are sent
after the ball lands. The colored marker is the chip prediction; the white
paddle remains fixed because the controller gate failed.

The five grids display all 160 measured integer match scores. The address grid
shows an actual 64-connection RAM record and its inserted/removed addresses.
The learning arm made 174 connection replacements during the filmed feedback;
the frozen arm made zero. These are logical records and sampled outputs, not
electrical spike timing or a physical die floorplan. The small native transaction
time excludes SSH, complete RAM capture and simulator time; it is not an energy
or system-latency benchmark.

## Evidence and reproduction

- `results/2026-10-23-pong-physics.json`: gates, comparisons and failed attempts.
- `results/2026-10-23-pong-physics.npz`: all 4,140 physical forwards across five
  sessions, scores, inputs, labels and complete per-forward RAM snapshots.
- `studies/20261023-pong-physics/PROTOCOL.json`: prospective banks and gates.
- `visual/pong-physics-data.json`: timestamped native packets and replay binding.
- `pong/audit_public.py`: checks public packets against the raw archive.

No hardware or BrainChip runtime is needed for the public recount:

```bash
python -m unittest tests.test_pong -v
python -m pong.audit_public
python tools/visualize_pong.py --video
```

Hardware sources are `pong/prepare.py`, `src/pong_stream.cpp`, `pong/serve.py`,
`pong/run.py`, `pong/live.py` and `pong/audit.py`. The compiler/runtime must be
supplied under their own licenses. Build using the existing `tools/build.sh`
environment with `AKD_LAB_BENCH=pong_stream`. Use a new short study envelope,
compile fixtures, freeze source/fixture/binary hashes, admit only the smoke
stage, then review the smoke before admitting a single matched pair. The owner
checks the shared lock, device ownership, frozen hashes, session/forward caps
and scientific cutoff. Individual processes have bounded deadlines and release
only their own device program. Closed studies cannot be resumed.

The failed 320-neuron compiler attempt selected FNP2 external-memory format,
which this decoder cannot inspect. It was preserved and reduced to 160 neurons
before hardware execution. A coarse encoder that aliased conflicting outcomes
was also rejected before hardware testing.

The next useful investigation is to separate fitting the new rule from
generalizing it, then test retention with court-B teaching suppressed. Additional
hardware throughput, full-chip saturation, power or competitive CPU efficiency
claims do not follow from this experiment.
