# Host involvement, bulk DMA and sparse event inputs

Direct user authorization: "Go build" following the sparse/resident/multi-learner
and real-image-work plan. Earlier studies remain closed. Maximum 24 physical
programming sessions and 80,000 native inputs; each owned process <=180 seconds.
Local AUTHORITY.json defines the prospective clock and cleanup deadline.

## Driver audit and scope

The locally available PCIe driver source exposes read/write/mmap/llseek and
interrupt-backed bulk-DMA completion. Its file operations expose no poll/ioctl
interface for inference completion. A kernel DMA completion is not an inference
completion interrupt. No driver, firmware, installed SDK, clock or voltage change.
The installed capability is admitted only after a scratch transfer smoke test.
Own adapter bounds continue to exclude the first 256 SRAM bytes used by eDMA.

## Stage gates

1. Owned scratch allocation: deterministic write/read equality through MMIO,
   then pread/pwrite bulk DMA, including 128/512/2048-byte aligned spans.
   Reject short/error transfers; no hidden retries or fallback. Close on error.
2. Full-score reference smoke, then DMA-read-only and DMA-read/write smokes;
   exact complete score vectors and final learning RAM must match Python.
3. Eight-job/nine-slot resident execution with busy completion polling versus
   bounded 50-us sleeps, preserving exact outputs and complete final RAM.
   Host CPU time is measured; sleeping longer is not itself an efficiency win.
4. Compiler-only dense/sparse versions of identical binary inputs, including
   sparse payload construction and prospective EPG/IoType checks. Admit only
   reviewed mapping and event-word encoding. Smoke matched streams first.
5. Three interleaved timing repeats of admitted representative paths, using
   identical inputs, labels and output contracts within each comparison.

The two-count reducer and full-score paths are separate endpoints. Bulk DMA may
improve the richer contract without discarding outputs; test that directly.
Sparse input encoding cost is inside the measured native transaction. Preloaded
resident buffers are labeled separately if used, never compared as equal work
to a CPU path that must encode fresh inputs. Include descriptor/input/readback
and host wait costs. Each confirmed stream has 256 inputs, CPU repeats 256.

Measure caller-thread CPU time, complete host transaction latency, per-stage
timing, actual MMIO/DMA bytes and calls, polling/sleep counts and IRQ deltas.
Thread CPU excludes unrelated threads and may exclude interrupt service on
another CPU; it is not total system CPU or electrical energy. Retain latency
tails and dynamic-clock platform metadata. No energy advantage without the
separate calibrated physical rig. Preserve every failed attempt.

Expansion to resident independent learners and actual image jobs gets a separate
prospective protocol, admission accounting and raw audit after this gate.
