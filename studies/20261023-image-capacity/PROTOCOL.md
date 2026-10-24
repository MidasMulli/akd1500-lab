# One capacity revision:3200 resident real-image exemplars

The480-exemplar study failed its frozen85% sealed accuracy floor: raw814/1000,
compacted817/1000. It is closed with zero hardware runs. This separately named
engineering revision tests whether inadequate exemplar coverage caused that
failure. It does not relabel the prior result as a pass. Exactly one revision;
no further bank/representation/model/seed search if this one fails.

Inherit the original pixel threshold128, centered32×32 binary images,2×2 spatial
OR, raw1024-byte host I/O, Hamming nearest-neighbor decision and original quality
targets. Both representations have the same source exemplars. All native neural
state is fixed; this is not an on-chip learning experiment.

Preserve the original48 exemplars per class in order. Seed98604 selects272 more
per class from unused indices in the same first10,000 training items, producing
320 per class/3200 total. Keep the prior200 development probes. Require each
arm>=80% and compacted no worse by more than3 points. Only then open a fresh
balanced1000-item subset from the official test data: seed98605, excluding all
1000 previously examined test indices. Require each>=85% and compacted no worse
by more than2 points. Save every old/new identity and prediction. No further
model changes after these observations.

Native graph revision is explicit: both raw and compact banks use one
Convolutional1×1 binary-weight layer with3200 filters, compiler-partitioned into
CNP1/CNP2 resources. Compact adds one fixed FNP3 spatial-OR encoder at(1,3,0).
Exclude that NPU from CNP choices in **both** arms. Return3200 overlap counts;
host chooses the first minimum foreground_count−2×overlap and its class (index
divided by320). The host never supplies intermediate image features to AKD.
This is not a fully on-chip argmin. Record/inspect actual resource mappings;
reject FNP2, external-memory, multipass or an alternative layer implementation.
No placement retry after a rejected mapping.

Use the original staged procedure, now on the3200-exemplar bank: compiler and
virtual checks;64-development-image physical smokes for raw then compact;
independent exact output/RAM/counter checks;1000-fresh-test-image screen raw then
compact; only if both median serial monitor service and hot-query wall time
improve>=5%, two additional temporal pairs with alternating path order. Maximum
8 sessions/8000 forwards/180 seconds per owned process and the unchanged parent
scientific/hard clocks. No physical access without quality and compiler passes.

Run exact-overlap, scalar packed nearest-neighbor and AArch64 NEON packed
nearest-neighbor CPU controls on the same Nano/core2, with the same byte inputs,
packing, pooling where applicable and final class decode included. Report all
CPU implementations and fixed compiler/source/binary hashes. Full batch timing
must include input upload, queue construction, completion, output drain and
decode; isolated release-to-completion is separate. Exact native overlap outputs
and host classifications must match reference on every query. Capture all
declared fixed NPU parameter regions before/after sessions.

Shared grayscale binarization, prototype loading and device programming are
excluded from hot-path comparisons and reported as such. No CPU or energy
advantage is presumed. DMA monitor scope is buffer service, not arithmetic-only
cycles. Card power is uncalibrated/unavailable. This is a standard real-image
recognition benchmark, not a novel application or all-chip saturation proof.

Stop and preserve any failed quality, compiler, ownership, state or cost gate.
No firmware/clock/installed-driver change, unsupported address sweep or killing
unowned jobs. Close admissions, release the device and publish bounded findings.
