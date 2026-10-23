# Identical-task transport comparison — October 23, 2026

All twelve runs replay exactly the original recorded 2,240-forward Pong learning
sequence. Every native score and complete final RAM match that original physical
capture and an independent CPU reconstruction. Three temporal repeats are
interleaved across four prospectively frozen configurations.

| Mode | Elapsed µs/input | Calling-thread CPU µs/input |
|---|---:|---:|
| Serial MMIO | 342.6 | 342.3 |
| DMA read/write | 452.9 | 89.3 |
| Queued DMA | 417.0 | 88.0 |
| Queued DMA + wait | 435.1 | 60.7 |
| Exact CPU neural-rule replay | about 11.2 | about 11.2 |

Queued values are batch-amortized elapsed cost, not individual event latency.
The CPU implements the same binary learning rule and returns identical outputs
and learned state. It includes input packing, output checksums and repeated
state initialization. Native costs exclude programming and correctness checking.

The measured MMIO fetch/completion section averages 315.3 µs/input, input writes
15.4 µs and submission 11.8 µs. Fetch includes waiting and output transport; it
does not isolate silicon computation. Enabling DMA in both directions raises
the input section to 204.3 µs. That observation motivates the separately frozen
read-only DMA follow-up; it is not a post hoc replacement of these primary cells.

DMA and queued waiting reduce calling-thread CPU from our MMIO baseline, but
all tested native paths still cost more than CPU replay on this small workload.
Calling-thread CPU is not total host/IRQ activity. No electrical energy or
silicon-peak advantage is established. The learning gate remains failed; these
recorded labels cannot establish batching of fresh causal game feedback.

Scratch transfers and all 12 replay cells execute correctly. **26,880** native
forwards are independently recounted; per-step RAM snapshots are not captured
in this cost study, while full final RAM and every score are exact. Admissions
are closed and failures of the separate scientific learning gate are preserved.
