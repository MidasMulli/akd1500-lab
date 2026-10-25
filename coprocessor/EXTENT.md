# Standard deployment and larger native workloads

The[October25 assessment](../studies/20261025-reference-scale/ASSESSMENT.md)
qualifies a pretrained reference plus32/64pixel native-learning graphs. CPU
OpenBLAS remains faster. The failed1,024channel and96pixel builds are preserved;
this architecture has not exhausted the silicon.

## Read-only reproduction

Requires Python/NumPy; the report also needs Matplotlib. No SDK/device opens:

```sh
python -m coprocessor.extent_audit audit
python -m unittest tests.test_reference_scale -v
python -m coprocessor.extent_report
python -m coprocessor.extent_visual --out /tmp/akd-extent-replay
```

For MP4 on macOS, add`--video`; the existing renderer uses FFmpeg VideoToolbox
for4K60 and a separate H.2641080p60 export. Public numeric arrays, source hashes,
reference outputs, exact RAM snapshots, timing order/exclusions and failures are
in`results/20261025-reference-scale.{json,npz}`. Vendor model weights, generated
SDK programs and the runtime library are not redistributed.

## Nano deployment / execution tooling

`deployment.py prepare MODEL.fbz` produces the deterministic synthetic input bank,
SDK software outputs, and virtual Minimal/AllNps programs. It never discovers a
physical device. Obtain the pinned official pretrained model separately:

```text
https://media.githubusercontent.com/media/Brainchip-Inc/brainchip_devhub/e7e4a57de706bfabb417e38824bee186e875a6ae/akida1/model_zoo/vww/pretrained_models/akidanet_vww_qat.fbz
SHA25690531d024b3359f836651748ff867aeaf0019aa0dd7067b9849fb5a3f5381b22
```

`scale_run.py` owns bounded standard-SDK reference admissions and the original
width screen.`extent_prepare.py OUTPUT SIDE SEED` builds only virtual fixtures.
`extent_run.py` admits the parameterized hardware/CPU actor, holds the shared lock,
checks device ownership, caps child duration/total work and freezes sources for
every process. The completed study is closed; these runners intentionally refuse
to reopen it. Use a newly authorized prospective study for additional hardware work.
Directly invoking the low-level actor bypasses those ownership/admission checks.

Build the AArch64 actor against the operator's installed, matching Akida2.19.3
runtime and FlatBuffers headers using`coprocessor/build_extent.sh`, supplying
AKIDA_ENGINE_ROOT, AKIDA_RUNTIME_LIBRARY and FLATBUFFERS_INCLUDE. NEON is required.
The optional BLAS control uses the installed NumPy ILP64 OpenBLAS0.3.27 shared
library through an explicit path and verified symbols/configuration. It does not
download, install, bundle or silently substitute a BLAS implementation. The Nano
runner currently records that installation's exact path/hash; port it explicitly
for a different environment before freezing a new study.

`extent_blas.hpp` implements three preallocated im2col+SGEMM stages, then pooling,
thresholds, feature packing and exact learning. Input transforms are timed; fixed
weight conversion is setup. The independent archive auditor recomputes signed
integer convolution and potentials from saved pre-event connection masks. It
checks complete CPU/native RAM, fixed CNP packets, teacher-class restrictions,
matched added/removed connections and query/frozen immutability.

Power remains a physical build:[parts and checkout](../power/FINISH.md).
