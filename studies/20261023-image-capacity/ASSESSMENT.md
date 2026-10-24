# Accuracy qualifies; the attempted native layer does not map

This one capacity revision preserved the original 480 exemplars and added 272
per class, for 3200 total. Both representations see the same images/exemplars.
Threshold128, padding, spatial-OR compaction and nearest-neighbor decision stay
fixed. No weight learning occurs; these are resident exemplar comparisons.

## Quality

Development: **182/200 raw, 183/200 compacted**. On a fresh balanced 1000-image
test that excludes every prior test image: **910/1000 raw, 900/1000 compacted**.
Both pass the original 85% floor, and the one-point loss stays within the frozen
two-point tolerance. This is a point-score engineering gate, not a statistical
non-inferiority conclusion. Different test subsets separate the failed small
study and the revision; their accuracy difference is not a matched capacity
treatment effect.

## Compiler barrier

Both 3200-filter Convolutional1×1 graphs pass exact virtual overlap/decision
checks on 16 images. Both then fail mapping with:

```
Cannot map layer 'resident_exemplar_overlap'. The maximum number of filters is 2048 (got 3200).
```

No program was launched, and no alternate graph or placement was substituted.
This establishes a compiler rejection for the attempted single layer. It does
not establish 2048 neurons/records as the chip's total physical capacity or
prove that multiple resident banks cannot solve the task. The physical study
is closed with **zero sessions and zero forwards**.

## Actual CPU control, measured after the compiler stop

Same Nano, core2, g++ 13.3.0, `-O3 -mcpu=native -std=c++17`. Three repetitions
of 1000 images per method/representation: **18,000 timed classifications**.
Methods rotate order across repetitions. Each hot path includes binary input
packing, CPU pooling when applicable and final class selection. File I/O,
initial resident prototype setup and shared grayscale thresholding are excluded.
The process reads the same 1024-byte binary images that would feed the AKD.

| CPU implementation | Raw median | Compacted median |
|---|---:|---:|
|Exact overlap vector + class decode|78.466µs|21.281µs|
|Scalar packed XOR/popcount nearest neighbor|87.363µs|25.601µs|
|AArch64 NEON packed nearest neighbor|48.0495µs|20.5445µs|

The supposedly simpler scalar alternative is slower here. The NEON alternative
is the stronger measured target. All methods agree on all decisions. The
independent auditor reconstructs both arms' full overlap matrices with packed
XOR distances and checks the captured CPU counts, every prediction, timings and
the preserved compiler failures. The nominal CPU frequency bounds read afterward
are both 1728MHz under schedutil; they are not a continuous frequency trace or a
claim of experimentally locked clocks. Source/binary hashes are retained.

## Scope and next route

This standard image benchmark makes the engineering target concrete: retain
useful accuracy and beat a roughly 20.5µs compacted CPU hot path under matched
boundaries. It provides no AKD speed, learning, energy or whole-chip utilization
result. The physical calibrated card-power rig remains unavailable.

A new native design must explicitly partition the bank and verify output
assembly. The previous generic multi-inbound FC merge already failed in the
metal study; do not assume concatenation works or retry the same unsupported
merge. Validate compiler-supported bank routing/output contracts or a reviewed
program-record construction before any new hardware admission. Keep all
exemplars and accuracy gates fixed; do not solve the compiler ceiling by silently
shrinking the qualified task. See `recognition/NEXT.md`.
