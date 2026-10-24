# Streaming prediction: native adaptation passes mechanics; skipping stays off

A frozen **T202** world model runs on the Nano GPU while one AKD learning FNP
predicts a bounded outcome from the arriving simulator images. Each query is
durably recorded before advancing five simulator steps. The subsequently received
RGB image supplies a teacher for the saved past input. No future image, physics
state, target or reversal flag enters the AKD encoder.

**The first matched pilot fails its prospective expansion gate. No inference
was skipped, and the other initialization seeds remain unrun.**

![Recorded stream, native responses and full matched results](../visual/streaming-prediction-4k.png)

[4K60 MP4](https://raw.githubusercontent.com/MidasMulli/akd1500-lab/main/visual/streaming-prediction-4k60.mp4) ·
[assessment](../studies/20261024-streaming-prediction/ASSESSMENT.md) ·
[prospective protocol](../studies/20261024-streaming-prediction/PROTOCOL.json) ·
[next gates](NEXT.md)

## What the matched comparison shows

Both native arms start with the same random initialization and complete native
training/query prefix. After that prefix, the learning arm receives delayed
teachers during adaptation; the frozen arm receives protected query teachers in
the same feedback slots. Its full parameter RAM remains unchanged. Every current,
previous, forecast and subsequent RGB frame matches exactly between the arms.

|Endpoint|Native learning|Trained-then-frozen AKD|CPU nearest neighbor|Frozen Nano T202|
|---|---:|---:|---:|---:|
|Changed stream, after feedback|49/69 (71.0%)|27/69 (39.1%)|61/69 (88.4%)|41/69 (59.4%)|
|Unchanged-context retention|56/74 (75.7%)|56/74 (75.7%)|70/74 (94.6%)|37/74 (50.0%)|

The 22/69 paired difference supports native adaptation in this pilot. Retention
accuracy stays equal to the frozen control. It does not establish a CPU advantage,
full RGB prediction, planning or generalization across initializations/devices.

The 75% changed-stream floor fails. Baseline target counts are `[0,28,20,3]`, so
the minimum class-coverage gate also fails. A fixed confidence rule selects 35
post-adaptation/retention queries and makes one error (2.86%), exceeding the 2%
limit. That is a shadow count, **not 35 actual skipped forecasts**.

## Actual data path

The Nano sees 224×224 RGB simulator observations and requested actions. T202
produces a five-step RGB forecast from its recurrent state. It is a frozen older
checkpoint, not a claim about the latest/best model or its CEM acting performance.
Its weights stay unchanged in all captures.

The host tracks the red disk in current/prior pixels and encodes position,
observed displacement, requested vertical command and a visible wall palette.
The AKD receives 64 active binary features in a 1,024-byte dense input. It does
not receive pixels, model latents or Nano forecasts. The target is one of four
equal vertical bands in the later observed image. These cheap inputs provide a
possible future skip path, but their existence alone does not qualify skipping.

Context A undergoes a disclosed host **command-routing reversal**. The original
simulator law and frozen model remain unchanged. Context B is unchanged. Actions
are prospectively generated, with no outcome-based action selection. This is a
controlled simulator stream; no camera or ESP sensor is involved.

One learning FNP `(1,3,0)` allocates 160 neuron records: 128 trainable records and
32 protected query records. This is not all-chip operation or a new communicating
multi-NPU network. Native response scores and full captured RAM match a compiled
exact-rule CPU reference and a separate Python reconstruction after every forward.
A compiled per-context 64-example nearest-neighbor control receives the same
delayed feedback. Pixel persistence and observed-motion controls are also saved.

## Verification and limits

Three device sessions (one smoke, one matched pair) produce **2,148 exact native
forwards and 2,148 full learning-RAM captures**. There are 533 model forecasts in
each matched arm, eight in the smoke, plus 27 in the model-only interface probe.
All model evaluations were retained. One initialization on one chip is not a
replicated or sealed confirmation result.

Prepared-input native transaction median is about 345 µs, versus about 12.3 µs
for the compiled task CPU control in the learning arm. Those figures exclude
encoding, complete RAM reads, durable journals, model execution and capture I/O.
The instrumented stream is not an end-to-end performance or energy benchmark.
No installed driver, runtime, firmware or clock was changed. The installed
PyTorch emits an Orin build-support warning; actual inference completes and
matched repeated forecasts are exact. The warning is preserved.

The public NPZ retains RGB frames, dense inputs, native outputs, timing samples
and lossless changed-word RAM deltas with each full snapshot's SHA256. Vendor
runtime, generated device programs, model weights and authorizations are external.

```bash
python -m streaming.archive audit-public
python -m unittest discover -s tests -p 'test_streaming.py' -v
```

`design.py` fixes the stream, encoders, seeds and gates. `nano.py` adapts the
unchanged model; `prepare.py` compiles without opening the AKD. `run.py` owns the
bounded physical capture; `audit.py` independently checks causal order, outputs,
memory and matched streams. `archive.py` publishes/recounts numeric evidence.
Physical reproduction requires a new authority, reviewed source/binary/fixture
freeze and staged admission. Completed studies remain closed.

## Visual

```bash
python -m streaming.visual --video
```

The 24-second 4K60 replay uses fixed phase-position samples, not samples selected
for correctness. It shows current RGB, the actual T202 forecast and the later
observed frame alongside measured native overlap responses and stored connection
changes. The four displayed groups contain 128 trainable records; query-protected
records are omitted. Dots are response scores, not electrical spike traces.
Each spoke denotes a stored input address with schematic geometry. The displayed
record is selected after capture for an actual edit, never fed into the acting
path. The movie is time-expanded recorded evidence, not a currently running UI.
