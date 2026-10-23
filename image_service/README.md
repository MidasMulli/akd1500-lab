# Real image-work adapter

The Nano does JPEG decoding, resizing and PNG export. The AKD receives compact
task/action/outcome fields, selects an operation using supplied native search
wiring, and updates transition and action-history connections from the actual
artifact outcome. It does not receive image pixels. This two-memory controller
is separate from the synthetic three-bank cascade.

Four owned CPU/GStreamer worker processes expose real outcomes. Outage studies
pause only these processes. Workers never invent success labels; Pillow checks
their actual pixel outputs. The CPU control uses exactly the same supplied
decision rule. It matches the native action/outcome sequences.

Build with `AKD_LAB_BENCH=image_service bash tools/build.sh`, with the normal
external runtime variables set. Start a new study with `tools/new_study.py`,
then run `tools/prepare_image.py studies/your-study` on the Nano. The preparation
uses the separately installed licensed Akida compiler; it does not open the
physical device. Inspect its checks before explicitly setting the study's
`ADMISSION.json` `watch` key to true. Never reopen a completed study.

From the study directory, a bounded pilot is:

```bash
AKD_IMAGE_WAIT_US=50 timeout --signal=INT --kill-after=15 180 \
  python service/app.py watch --name pilot --inbox inputs/71011 \
  --outbox outbox --max-jobs 2 --max-seconds 60
python verify_native.py pilot
python verify_service.py runs/pilot
```

Run one hardware owner at a time. The client checks the deadline, study state,
source manifest, session cap, common device lock and device occupancy. The
outer timeout bounds workers and their parent process group. Do not invoke
outage/recovery series without a reviewed protocol and its `confirm` admission.

`AKD_IMAGE_WAIT_US=20000` retains the earlier demonstration's deliberate 20 ms
protected no-I/O interval. The new 50 µs path sleeps, then reads completion
status until finished. The auditor permits those reads and still rejects every
host write inside the released queue. This removes an artificial guard wait;
it is not a silicon-compute speedup. Full MMIO tracing and neural snapshots
remain instrumentation overhead. No service latency or energy advantage over
the CPU is established.

Native memory lasts within a running session. Every fresh program starts from
the fixture; this is not power-loss durability or resumed hidden-state proof.
Only our source is packaged here. Generated vendor program images and runtime
sources are excluded. `tools/audit_image_copy.py` checks an unchanged copied
study with explicit path relocation for the source artifacts.
