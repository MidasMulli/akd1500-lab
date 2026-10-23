# Read-only DMA follow-up — October 23, 2026

This is a targeted, prospectively frozen follow-up after the primary four-mode
study found expensive DMA writes for 1,024-byte inputs. It retains the same
recorded Pong stream, program, labels and exact learning endpoint. Each mode
has three interleaved temporal repeats on one chip.

| Mode | Elapsed µs/input, mean [range] | Calling-thread CPU µs/input |
|---|---:|---:|
| Serial read-only DMA | 254.8 [233.3–295.5] | 83.0 |
| Queued read-only DMA + wait | 256.0 [235.0–297.3] | 54.7 |
| Exact CPU replay | about 11.2 | about 11.2 |

Read-only DMA improves elapsed cost versus enabling DMA in both directions.
The queued waiting path reduces calling-thread CPU by about **84%** from the
original MMIO path's 342.3 µs/input. It still uses about 4.9 times the caller CPU
of exact CPU replay on this workload. These are measured software/runtime/card
paths, not silicon-only costs. Caller CPU excludes system/IRQ activity.

Queued elapsed/input is batch-amortized cost, not individual event response
latency. The variable elapsed times are preserved; no cause is assigned without
a controlled test. Input generation, programming, independent verification and
full learning-RAM capture are outside native timing. CPU replay includes input
packing, checksums and repeated fresh-state initialization. No electrical energy
measurement or chip-peak conclusion follows.

The scratch smoke and six replay sessions pass. Every score and full final RAM
match the original sequential physical stream and independent CPU reconstruction:
**13,440 native forwards**. This does not reopen the failed prediction gate,
establish fresh-game batching, or demonstrate a favorable CPU differentiation.
Admissions are closed and the device is released.
