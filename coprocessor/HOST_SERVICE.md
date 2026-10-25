# Resident learning with bounded host service

The[assessment](../studies/20261025-host-service/ASSESSMENT.md) separates lower
caller CPU from latency, foreground throughput and unmeasured energy. The study
is closed; its runners refuse additional admissions. Use a newly authorized,
prospectively bounded study to change settings or run more hardware cells.

## Portable review, no SDK/card required

```sh
python -m coprocessor.service_archive audit
python -m coprocessor.service_report
python -m unittest tests.test_host_service tests.test_host_window -v
python -m coprocessor.service_visual --out /tmp/host-service-replay
```

Python/NumPy are required; plots use Matplotlib and the replay uses Pillow.
For 4K60 and 1080p60 H.264 exports on macOS, add`--video`and install FFmpeg.
The manifest binds results/20261025-host-service.tar.gz to each original numeric
file and receipt. It includes all fixtures' numeric inputs/weights, exact traces,
failed setup and pre-device/offline repairs. No licensed runtime, generated
vendor program or T202 checkpoint is bundled. Do not repack the archive casually:
the visual binds its exact SHA256 and the raw sources it depicts.

The auditor extracts only listed regular files beneath a temporary directory,
checks their bytes, independently reconstructs spatial features and the learning
selector, validates complete captured states/outputs, and reruns timing/barrier
checks. It cannot certify electrical energy, physical neuron concurrency or
unobserved states between a batch's captured boundaries.

## Tooling

`service_prepare.py`creates virtual potential/binary-output fixtures from the
unchanged resident graph.`build_service.sh`builds the AArch64 actor against
operator-supplied Akida 2.19.3 headers/static runtime and FlatBuffers headers.
It uses the exact ILP64 OpenBLAS shared library already installed on the Nano;
source and external binary hashes are frozen before execution.

`service_run.py`enforces source/fixture hashes, strong CPU controls, sequential
admission gates, shared lock, free-device checks, duration/session/forward caps
and deadline. It preserves every owner/stdout/stderr/receipt and halts on failure.
`service_confirm.py`implements the prospectively selected rotated confirmation.

`window_source.py`generates a separate source variant from the exact service
parent. It adds shared-start pacing and an audit-release barrier without editing
the original qualified source. Existing FNP2 window_run/build_window tools are
separate and unchanged. New tooling uses`host_window_run.py`and
`build_host_window.sh`to avoid collisions.

`window_model.py`runs a separately loaded, checked T202 instance without touching
the deployment/server. Eight exact warmups precede READY. Every completed output
hash and the final parameter digest are checked. Native/CPU actors use identical
event order; no observer/reference work may execute until the foreground ends.
The installed runtime warning and its bounded functional validation are retained.

The 64-event contract remains eight prefix queries, 32 teaching events, 16 repeated
queries and eight novel queries. `mask`returns potential>24 per record; it is
not an interchangeable consumer contract with full potentials. Mode changes,
input copy, enqueue, completion and output copy are timed. Diagnostic work is
excluded. Batch cost is measured; per-image amortization is derived. Per-call
process CPU excludes between-call worker spin; common-window counters preserve
that additional cost separately.

Power remains a physical build: [required hardware and checkout](../power/FINISH.md).
