# Real-image compaction: quality passes, native mapping stops

This extends the synthetic redundant-input experiment to **real MNIST images**.
It is a standard recognition benchmark, not a novel application. No native
learning or new physical AKD timing was measured in this campaign.

The task keeps a bank of binary image exemplars resident, compares a query with
every exemplar, then picks the nearest image's class. The compact representation
uses 2×2 spatial OR on the chip's intended path. Native graphs would receive the
same 1024-byte binary input and return overlap counts; final nearest-neighbor
selection remains on the host. This is not a fully on-chip classifier decision.

* The original 480-exemplar bank failed its frozen 85% quality floor: 81.4% raw,
  81.7% compacted. That study is closed and preserved.
* One larger-bank revision retains those exemplars and extends to 3200. A fresh,
  disjoint test reaches 91% raw and 90% compacted, passing the unchanged quality
  floor/tolerance. It is not a matched estimate of the capacity effect because
  the two studies use different test subsets.
* Both intended 3200-filter native layers fail the compiler's 2048-filter ceiling.
  No alternate implementation was substituted and **no hardware run occurred**.
  A per-layer compiler ceiling is not a measured total-chip capacity limit.
* CPU controls agree exactly. The compacted NEON nearest-neighbor hot path takes
  **20.5445µs median**, including pooling, packing and decode. Raw NEON takes
  48.0495µs. The exact-network and scalar alternatives are also reported.

[Small-study failure](../studies/20261023-image-compaction/ASSESSMENT.md) ·
[larger-bank assessment](../studies/20261023-image-capacity/ASSESSMENT.md) ·
[public numerical evidence](../results/2026-10-23-image-capacity.json) ·
[source data/provenance](SOURCES.md) ·
[next executable design gates](NEXT.md)

## Recount without hardware

```bash
python -m pip install numpy
python -m recognition.archive audit-public
```

The NPZ contains selected binary images/exemplars, original source identities,
captured CPU outputs and timings. The auditor uses independently implemented
spatial pooling and packed-XOR Hamming distances rather than the producer's
integer-matrix overlap reference. No downloads or vendor runtime are needed
for this recount. Dataset replay from the original IDX sources remains available
through the data loader; completed study directories refuse overwrite.

`data.py` defines the failed small-bank study; `capacity.py` the single frozen
revision; `prepare.py` the failed compiler/virtual attempt; `cpu_bench.cpp` the
actual three CPU implementations; `archive.py` the publication and independent
checks. Compiler preparation requires the separately licensed Akida package.
No physical launcher was authored after the compiler gate failed.

The source-bound 4K figure shows real image transformations, quality results,
actual CPU times and the compiler stop. It does not depict live AKD activity.
