# PCIe co-processor environment

**The Nano is an appropriate documented host.** BrainChip positions PCIe as
an offload interface for ARM/x86/RISC-V application processors and M.2 as a
drop-in accelerator for fanless industrial/commercial systems. See the
[production announcement](https://brainchip.com/press/brainchip-announces-commercial-availability-and-production-shipments-of-akd1500-neuromorphic-processors/),
[M.2 announcement](https://brainchip.com/press/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/)
and [current product](https://shop.brainchipinc.com/products/akd1500-edge-ai-co-processor).
No task-specific benefit follows merely from that packaging.

We built and physically checked the documented neural path:

```
Nano raw RGB / class labels
       | PCIe Gen2 x2, observed on this card
       v
HRC image-to-events convolution (8-bit weights)
       | native events
       v
CNP spatial convolution (4-bit weights, binary output)
       | 512 binary features, native spatial-to-FC order
       v
FNP3 native binary learner (128 records, 32 connections each)
       | 128 integer potentials returned over PCIe
       v
Host consumer
```

All three layers occupy **one resident compiler pass**. Native acting code
does no host feature extraction. Every score and full captured learner state
matches independent optimized C++ and separate Python replay. These are seeded
characterization images/kernels, not a trained vision application.

## Evidence

45 owned physical sessions, including one zero-forward transport copy check:
**3,872 forwards, 1,016 full learner-RAM snapshots**. Twelve CPU control streams
each have five fresh-state repetitions, 5,760 CPU sample executions. All pass.

Two compiler-only failed attempts are retained: wrong convolution convention,
then wrong spatial-to-FC ordering. No device forward occurred in either attempt.

| Widest fixed workload, queued eight images | Frozen elapsed µs/image | Learning elapsed µs/image | Frozen caller CPU µs/image | Learning caller CPU µs/image |
|---|---:|---:|---:|---:|
| Optimized CPU dense NEON | 247.7 | 248.4 | 247.0 | 247.8 |
| AKD mapped I/O | 290.3 | 446.8 | 289.9 | 446.1 |
| AKD kernel DMA read | 283.7 | 440.0 | 133.1 | 289.4 |
| AKD kernel DMA read + write | 707.4 | 707.3 | 99.1 | 99.2 |
| AKD kernel DMA write | 762.1 | 763.5 | 303.1 | 303.6 |

Native values are medians of three temporal run medians on one chip and
initialization. CPU numbers are medians after 16 warmup images across five
fresh-state runs. A second verified CPU implementation skips zero-valued input
contributions; it is slower here and retained. Native and CPU timing includes
all work from prepared RGB bytes to returned potentials/update, including
native transfer/enqueue/fetch. It excludes programming/preallocation, diagnostic
RAM reads and independent verification. Surrounding diagnostics still perturb
the stream. Calling CPU time is not kernel/whole-host CPU or electrical energy.

**No overall speed or energy win is established.** DMA reduces caller activity
at a substantial latency tradeoff; it does not make the tested learner faster
than CPU. CPU thread pinned to Nano core 2; clocks were not changed. CPU controls
are optimized exact-math implementations, not a claim of global CPU optimality.

![Actual resource endpoints and measured costs](../visual/pcie-fabric-4k.png)

## Resource map and boundaries

[FABRIC_MAP.md](FABRIC_MAP.md) separates documented architecture, SDK descriptors,
physical observations, compiler-format boundaries and remaining unknowns.
The [source manifest](DOCUMENTS.json) records reviewed PDF versions, URLs and
hashes; vendor PDFs and licensed runtime code are not redistributed.

## Reproduce without hardware

```bash
python -m coprocessor.archive audit-public
python -m unittest tests.test_pcie_fabric -v
python -m coprocessor.visual
```

The JSON/NPZ pair under `results/2026-10-24-pcie-fabric` contains original RGB,
weights/thresholds, axon features, initial/final RAM, complete boundary states,
scores and all timing samples. The replay independently recomputes both
convolutions, spatial ordering and learning state; it rejects missing controls.

## Hardware reproduction

Requires separately obtained Akida SDK 2.19.3, matching runtime/FlatBuffers
headers, existing Linux PCIe driver and ARM NEON compiler target. Supply
`AKIDA_ENGINE_ROOT`, `FLATBUFFERS_INCLUDE`, and `AKIDA_RUNTIME_LIBRARY` to
`bash coprocessor/build.sh` / `bash coprocessor/build_transport.sh`.

The saved campaigns are closed and runners refuse new admissions. Start a
fresh study with explicit authority, caps/deadlines, seed fixtures and a source
freeze. Follow the saved staged protocols; do not overwrite an old run or
bypass a failed gate. The existing caller-buffer lifetime and nine-slot/eight-
image queue protections remain in place. No driver, firmware, PLL or unknown
register writes are needed. `coprocessor.inventory` is sysfs/descriptor-only.

The code package is a characterization environment. Full FNP2 host-memory
learning and multi-pass operation remain outside its qualified execution path.
