# Native streaming adaptation observed; prediction and skipping gates fail

Closed October 24, 2026. Three owned device sessions complete **2,148 forwards and
2,148 full native RAM captures**. One initialization, one matched native pair,
533 queries per arm, plus an eight-query native smoke. The model-only probe adds
27 forecasts without AKD access. No model forecast was skipped.

The isolated T202 model checkpoint SHA256 is
`3342f73cf939f689abfb1189faf3580719acbf05ca2d439001ee13887ea5bc65`.
All frozen-model weights remain unchanged; matched-arm RGB observations,
forecasts, outcomes and AKD inputs match exactly. The existing on-demand server
and deployed files are unchanged.

|Changed-stream outcome, held from adaptation teaching|Correct / total|
|---|---:|
|Native delayed-feedback learning|49/69 (71.0%)|
|Same trained prefix, then frozen native weights|27/69 (39.1%)|
|Compiled delayed-feedback nearest-neighbor CPU|61/69 (88.4%)|
|Pixel persistence|41/69 (59.4%)|
|Observed-velocity extrapolation|37/69 (53.6%)|
|Frozen T202 five-step RGB forecast, same band tracker|41/69 (59.4%)|

Unchanged-context retention is 56/74 in both native arms. The native pair has
identical initial weights, trained RAM and prefix outputs. Frozen post-prefix
parameter RAM never changes. The learning arm records 4,862 post-prefix
connection-bit edits. Every native potential and complete captured memory state
matches independent exact-rule reconstruction.

Each query is durably committed before the environment advances. The later RGB
image determines its teacher; that teacher is applied to the saved past input.
Actual future state, simulator internals and the reversal indicator never enter
the binary encoder. The host supplies the teacher and timing; native hardware
selects eligible neuron records and connection changes.

## Prospective stop

The 75% changed-stream quality floor fails. Baseline target counts are
`[0,28,20,3]`; the requirement for at least two examples in every band also fails.
Learning-versus-frozen gain, persistence gain and retention gates pass. The pilot
is useful evidence of native adaptation, not four-class qualification.

The predeclared confidence rule (score at least 56/64 and margin at least 8)
selects 35 post-adaptation/retention queries and makes one error: 2.86%, exceeding
the 2% limit. The 95% changed-stream skipping floor also fails. **Skipping stays
disabled, and seeds 105102/105103 are unrun.** No threshold, seed, target or
representation replacement is made inside this study.

## Bounds

The stream uses the existing simulator renderer, not a camera or ESP sensor.
The changed context has a disclosed host actuator-command reversal; neither
world-model weights nor the simulator's physical law is changed. Cheap,
host-supplied pixel/motion/action codes feed a single plastic FNP `(1,3,0)` with
160 records. This is not raw-pixel learning, full-chip learning, autonomous
teaching or a communicating multi-NPU network.

The prepared-input native transaction median is 345.45 µs in the learning arm;
the compiled exact-rule CPU is 10.30 µs and the task nearest-neighbor CPU is
12.26 µs. Encoding, packing, full RAM capture, journals and the model are excluded
from these primitive timers. Instrumented full-run wall time is not an efficiency
benchmark. No CPU speed or calibrated energy advantage is established.

The installed model environment emits an Orin build-support warning. Actual
deterministic inference runs, weights stay unchanged and all paired forecasts
repeat exactly. No environment upgrade or installed driver/runtime/firmware/
clock change is performed. The older T202 model's weak band score is not evidence
about the latest or best world-model checkpoint.

A relative-path manifest-authoring error and its subsequent pre-admission
rejection are preserved with zero physical forwards. A presentation-only missing
import is corrected separately. All three admitted physical sessions pass
execution and independent verification; the scientific quality gate fails.

Full numeric frames/inputs/outputs and lossless RAM deltas are public. Recount:

```bash
python -m streaming.archive audit-public
```

The source-bound 4K60 replay shows actual contemporaneous model/chip capture,
with fixed phase-position sample selection and time-expanded playback. Response
dots and connection spokes are measured data with schematic geometry, not
electrical spike telemetry. The practical-advantage X-post gate remains unmet.
