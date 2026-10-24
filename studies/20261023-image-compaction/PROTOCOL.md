# Real-image recognition with native spatial compaction

Direct operator Go after the bounded synthetic graph result. This is an
engineering benchmark on MNIST handwritten images, not a novelty, state-of-art
accuracy or substrate advantage claim. Frozen exemplars isolate representation
and execution cost. No native learning occurs in this study.

## Data and decisions frozen before inspection

Download original MNIST IDX files from the CVDF-maintained dataset mirror,
retaining source hashes. Threshold pixels at128; center28×28 in a zero-padded
32×32 binary image. Exactly one compact representation: 2×2 spatial OR, producing
16×16 binary features. The same transformation is applied to exemplars.
No threshold/radius/model search or alternate compaction after results.

Seed98601 selects48 training images per digit from the first10,000 training
indices, in class order, for480 resident exemplars. Seed98602 selects20 per
digit from the remaining50,000 training images for200 development probes.
Only after development passes, seed98603 selects100 per digit from the10,000
official test images for1000 sealed probes. Save all selected indices.

Both native graphs receive exactly1024 binary uint8 bytes and emit480 overlap
counts. Each exemplar's foreground count is fixed metadata. Host decoding picks
the first exemplar minimizing foreground_count−2×overlap, then its digit label.
The query foreground count cancels across exemplars; the decoder needs no true
label or representation on the host. This is exact Hamming nearest-neighbor
classification within each representation, not a native on-chip argmin.

Development gate: each representation>=80% correct and compacted no worse than
raw by more than3 percentage points. Sealed qualification: each>=85% correct,
compacted no worse by more than2 points. Preserve/stop if either fails; do not
tune or replace seeds. Agreement with labels is distinct from exact hardware
agreement with each arm's own numerical reference.

## Native and CPU contract

Raw: one fixed FNP3 detector bank using1024-input records. Compact: fixed FNP3
spatial-OR encoder1024→256, then256-input detector records. Hold detector
placement FNP3(2,1,0) fixed; encoder FNP3(1,3,0). Reject FNP2, external-memory or
multipass substitutions. Count the encoder's parameters and service cost.
All native intermediate features remain on-chip. Complete declared fixed
parameter-region readbacks before/after each physical session must match.

Three CPU implementations run on the same Nano, pinned to core2, from the same
1024-byte binary input: exact overlap-count output, scalar packed XOR/popcount
nearest neighbor, and AArch64 NEON packed nearest neighbor when supported.
Include input packing, CPU spatial pooling when used, and final class selection.
Report all implementations, not just the slow exact-network comparator. Fixed
weights/exemplars remain resident; programming/loading are separately excluded
from both hot-path timings. Common grayscale thresholding is excluded in both.
Compiler flags, source/binary hashes and per-image predictions must be retained.

## Stages and stop rules

CPU/data qualification and compiler/virtual verification precede physical access.
Freeze owned source, binary, data identities, fixtures and bounded authority.
Exclusive device lock plus occupancy check; no second owner, unowned process
termination or installed firmware/driver/clock modification.

1. Raw and compact smoke: first64 development images,64 serial queries plus
   three paired8-job serialized/queued suffixes. Inspect all outputs, host-decoded
   classes, before/after parameter memory, idle/config counters and queue outputs.
2. Only after exact checks pass, raw and compact screen on1000 sealed images,
   with three paired8-job suffixes. Require every suffix image nonzero, and
   report its actual encoder/detector activity. Dense upload is unchanged.
3. Admit two additional temporal pairs, alternating path order, only if compact
   median serial DMA service **and** complete hot-query wall time (upload,
   enqueue, fetch, host decode) each improve>=5% in the screen. Otherwise stop
   and preserve the losing native comparison; no larger bank or alternate model.

Maximum8 sessions/8000 forwards,180 seconds per owned process, and the recorded
scientific/hard clocks. Stop on any correctness, memory, ownership or time failure.
Record process cleanup and close admissions. Independently recount public raw
numerical evidence before publication. Preserve all failed attempts.

DMA counters measure buffer service, not isolated arithmetic or electrical power.
Queued release timing excludes preload/descriptor construction/result drainage;
also report a full queued-batch wall timer including those costs. No zero-input
burst proxy or card-only versus host-total energy comparison. The physical
calibrated card-power rig is unavailable: do not infer energy savings from cycles.

MNIST is a standard real-image benchmark, not a deployed user application. This
small resident bank does not saturate the32-NPU chip or prove a native advantage.
