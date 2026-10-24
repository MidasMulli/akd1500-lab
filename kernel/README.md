# Native execution rewrite and learning-policy screens

This package asks what we can actually improve on the AKD1500. It separates a
supported native graph rewrite from a replacement for the internal learning
engine. Only the graph rewrite qualified in this campaign.

## Measured native graph compaction

The original graph gives each detector a 512-input record. Only 128 positions
carry unique information in the declared four-identical-copy input domain. The
rewrite uses an additional fixed native NPU to reduce those copies to 128 bits
before the detector bank. The host still uploads the same 512 uint8 values and
receives the same two integer counts. No intermediate activation returns to the
host. Both graphs have 32 selected connections per detector and the same latent
weights and output rule.

Across 16 physical sessions and 2,624 forwards, every output matched the matrix
reference and every captured fixed parameter region stayed exact. At 512
detectors, median half-active DMA service fell from 16,626 to 15,642 monitor
cycles (5.92%). Captured parameter payload fell from 41,120 to 26,784 bytes
(34.86%), including the extra encoder. Three fresh-program temporal sessions
reproduced the same median cycle counts; they are not three physical chips.

The tradeoff is substantial: the extra NPU made 256-detector half-active service
18.17% more expensive, and fully active 512-detector service 5.70% more expensive.
Eight queued zero-activity inputs used 21.05% fewer monitor cycles at width 512.
That burst excludes input preload, descriptor construction and output drainage;
it is not nonzero-activity throughput or end-to-end application latency.

[Frozen protocol](../studies/20261023-native-compaction/PROTOCOL.md) ·
[full assessment](../studies/20261023-native-compaction/ASSESSMENT.md) ·
[raw numerical archive](../results/2026-10-23-native-compaction.npz) ·
[derived assessment](../results/2026-10-23-compaction-assessment.json) ·
[physical-source provenance](../results/2026-10-23-compaction-provenance.json)

```bash
python -m kernel.experiment audit-public
python -m kernel.report
python -m tools.visualize_native_compaction
```

These commands do not access hardware. `kernel.prepare` requires the external
vendor compiler; hardware sessions require a separate bounded authority, source
freeze and explicit staged admission. The saved study is closed.

## Learning-policy candidates did not qualify

`recipe/` forecasts policies around the previously reconstructed native update
rule: delayed feedback, confidence/error teaching gates, and protected logical
populations with equal total connection resources. These are CPU forecasts, not
new physical learning results and not arbitrary on-chip update instructions.
The host would decide when to supply feedback; the native engine would perform
the connection updates.

Both initial policies failed quality gates. A conservative development repair
preserved quality but missed its original cost target. A separately frozen
four-seed validation targeted a smaller saving in total query-plus-teaching jobs;
it still failed quality on two seeds. No native learning run was admitted.

[Initial failure](../studies/20261023-native-recipe/ASSESSMENT.md) ·
[development repair](../studies/20261023-native-recipe-r1/ASSESSMENT.md) ·
[sealed validation](../studies/20261023-native-policy-validation/ASSESSMENT.md)

## What remains open

The native result supports representation-aware graph construction. It does not
establish a useful application, a CPU advantage, energy savings, sparse ingress,
an improved native learning algorithm or arbitrary NPU programmability.
[The programmability map](PROGRAMMABILITY.md) identifies the interfaces reviewed
and the boundary we have not crossed.

The next useful test would find real redundant/event features in a resident task,
freeze an exact or accuracy-bounded representation, and compare total completion
and calibrated energy against both the original AKD graph and an optimized CPU
task solution. Extend burst testing to nonzero activity before generalizing its
21% service reduction. Do not expand the failed learning-policy search with
replacement seeds.
