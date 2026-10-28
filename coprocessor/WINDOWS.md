# Native class-window learning over a larger DDR-backed model

**A bounded workaround passes; the original full-bank learning path still fails.**
One AKD1500 FNP2 operates on an 11,264-record model with 512 binary inputs and
32 selected connections per record, stored in Nano host DDR.
Full inference sees the entire bank. During teaching, the host selects a
5,632-record class view using the already supplied teacher label. The chip
chooses the winning record and changes its connections; the host does not edit
live weights. The same full model then performs read-only recall.

![Recorded direct failure and guarded native class window](../visual/fnp2-window.png)

[4K60 replay](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/fnp2-window-4k60.mp4) ·
[1080p60 download](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/fnp2-window-1080p60.mp4) ·
[post draft](../visual/fnp2-window-X-DRAFT.txt)

## What changed and what passed

The direct control uniquely selects record 8849 but changes record 2295.
With the class-window view, record 8849 updates exactly as the full-model C++
and independent Python references predict. Its subsequent full-model match
response is 32/32; the matched frozen control remains 31/32. This response is a
connection-match count, not a classification accuracy percentage.

The larger sealed stage tests eight selected addresses in each of three fresh
weight initializations: 107101, 107301 and 107401. Targets include 0, 256, 5000,
5631, 6600, 8849, 10000 and the final record, 11263. All **24 selected teaching
updates** and full-model recalls match; every complete bank snapshot is checked.
Every frozen probe remains at 31/32, and native learned recall reaches 32/32.
These probes have a unique eligible winner, one missing connection and one
novel connection. They do not qualify arbitrary learning streams or every
allocated record.

Window study totals, including the pilot, provenance checks and failed direct
control: **12 physical sessions, 101 forwards, 110 complete bank captures**.
Six optimized C++ CPU controls execute five fresh-state repeats each, for
480 model forwards. Zero host writes to the live record bank are observed
after its initial loading. The driver rejects any such attempt.

## Why the workaround is grounded

The source audit and native interventions narrow the original failure:

| Intervention | Observed result |
|---|---|
| Stock SDK, 512 inputs / 8192 records | The same record6600→record46 failure reproduces |
| Stock allocator provenance | Actual registered records at 0xC0000040 in host DDR; complete mapped and API bytes agree |
| 1024 inputs / 4096 records | Record3600 updates correctly; record3700 changes record59; metadata separation remains 131072 words |
| Same record3640, connection word crossing offset131072 | Learning is exact; a blanket modulo on every write address is contradicted |
| Adjacent record3641 | Wrong changes appear in record0; metadata separation again 131072 words |

This points toward record-base arithmetic in the configured learning path;
it does not establish the internal circuit, register field, universal capacity
limit or a manufacturer silicon erratum. Several earlier explanatory models
remain rejected. The original path remains failed in all public records.

The native window reduces the index relative to its registered base. Global
record8849 becomes local record3217 in class1. A smaller compiled view accesses
the **same stored records**, so the connection updates are visible when the
full model resumes. No copying back of CPU-generated learned weights occurs.

## Address ownership and observation

The window runtime uses disjoint address domains:

- Host DDR: one kernel-accepted 4 MiB mapping, with immutable program data and
  a protected live record array in separate bounded regions.
- Card SRAM: runtime control descriptors, inputs and outputs, managed by the
  existing runtime allocator.
- Known register interface: the existing FNP2 base register, configured by
  `program_external_data` and read back to verify the selected view.

The only serialized relocation changes a compiler-generated FNP2 track offset
before execution; neither the connection data nor the learning rule is patched.
Both API memory and direct record views are checked before each view runs.
The driver observes every host write through the runtime and rejects overlap
with the live record bank. This is an experiment guard, not hardware access
control against arbitrary kernel writers. One process owns the device.

A separate manual external-data placement attempt fails the frozen state gate:
it is preserved and **cannot establish a learning result**. It did not reserve
space against the default runtime allocator; output-pattern overlap is a
candidate explanation. The qualified workaround instead separates allocators
by construction and verifies the full bank. A failed zero-forward observation
attempt, caused by an in-process handle-release check, is also retained.

## Reproduction and scope

```bash
python -m coprocessor.window_archive audit-public
python -m unittest tests.test_native_windows -v
python -m coprocessor.window_visual --video
```

The public JSON/NPZ archive stores inputs, labels, all responses, full initial
and final RAM, lossless deltas of every captured full bank, original raw-state
SHA256 checks, guard counts, source identities and failed prefixes. It also
replays the stock SDK, record-width, same-record and adjacent-record diagnostics.
The seven new tests reject missing controls, hidden failures, RAM tampering and
nonzero host weight-write attempts.

Hardware compilation uses `bash coprocessor/build_window.sh` with separately
obtained Akida 2.19.3 runtime sources, matching FlatBuffers headers and runtime
archive. The existing studies are closed: reproduction requires a new bounded
study and source/fixture freeze. Licensed runtime sources and SDK-generated
program binaries are not redistributed. No driver, firmware or clocks changed.

This is **host-scheduled native learning**, using one FNP2 and external volatile
DDR. It is not autonomous multi-NPU scheduling, power-loss durability, a trained
vision application, a whole-chip maximum, or a CPU/energy advantage. Arbitrary
multi-swap learning across view changes needs a fresh hidden-selector-state
study. The film uses recorded responses and RAM changes; layout and motion are
illustrative and the matched recordings are separate runs on one chip.
