# Native single-class context allocation

Thirty-two initially random plastic records share a single training class. The
AKD chooses the eligible record to update. Eight read-only helper records carry
raw history through a fixed relay; Nano provides only the current symbol and
training schedule. No history ID, context lookup or selected-neuron label enters
the acting path. The fixed four-symbol/third-order feature representation is
engineered; only record allocation and connection changes are learned.

Five initializations recall16/16 histories each, retain4/4 original record IDs,
and pass128/128 held-out windows each. Strong CPU dictionary also passes. This
is a component qualification, not an autonomous world model or efficiency result.

## Numeric-only audit (no SDK or hardware required)

Install NumPy, then from the repository root:

```sh
python -m discovery.archive audit
python -m discovery.audit_mutations
python -m discovery.screen
```

`results/20261024-context-discovery.{json,npz}` holds saved input/output packets,
per-event learning RAM, descriptors, fixed-bank checks and CPU traces. Generated
SDK program images, licensed runtime source and hardware build products are not
published. Failed and superseded traces remain distinguished in the manifest.
Seven rehashed semantic mutations must be rejected. The separate screen is an
exploratory rejected representation comparison, with different input densities.

## Source roles

- `model.py`: fixed feature/relay matrices, seeded weights, raw-symbol schedule,
  independently reconstructed bit-set learning reference and raw CPU dictionary.
- `prepare.py`: virtual/compiler-only graph preparation on a virtual AKD1500;
  never discovers or opens a physical device.
- `bench.cpp`: owned native instrument and independently packed CPU implementation;
  reads previous native output directly, writes only current-symbol channels,
  captures packets/descriptors/full learning RAM, and unprograms its own model.
- `run.py`: private authority/deadline, exclusive device lock, source freeze,
  finite sessions/forwards, staged gates and preservation of failed attempts.
- `unseen.py`: append-only, sealed new-order fixtures after main qualification.
- `archive.py`: portable source-derived verification of measured numeric traces.
- `visual.py`: 4K60 source-bound recorded replay. Motion and coordinates schematic;
  native returned activity and RAM edits measured; no electrical spike claim.

For a separately licensed SDK/runtime environment, use `python -m discovery.prepare
<new-private-fixture-folder> <seed>` for compiler-only checks. `sh discovery/build.sh`
requires `AKIDA_ENGINE_ROOT`, `AKIDA_RUNTIME_LIBRARY`, and `FLATBUFFERS_INCLUDE` pointing
to separately supplied matching files. Physical runs additionally require an explicit
private bounded AUTHORITY.json and STATE.json for the study, frozen sources/fixtures,
a free device and the research lock. The qualified study is closed; do not reopen its
old authority for a new experiment. Create a prospective study and new bounds.

[Assessment and failures](../studies/20261024-context-discovery/ASSESSMENT.md) ·
[1080p60 replay](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/context-discovery-1080p60.mp4) ·
[4K60 replay](https://github.com/MidasMulli/akd1500-lab/releases/download/media-20261025/context-discovery-4k60.mp4)
