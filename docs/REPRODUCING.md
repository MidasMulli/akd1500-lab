# Evidence, replay and reproduction

The source checkout provides the Explorer and static recorded demos. Evidence bundles are immutable public editions of the preserved numeric captures; scientific bytes retain their hashes. Operator authority/session metadata, draft posts and linked SDK binaries are excluded. Original private review packages remain preserved separately.

## Versioned research downloads

| Artifact | Size | SHA256 |
|---|---:|---|
| [20261027-silhouette-managed-memory-evidence.zip](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/20261027-silhouette-managed-memory-evidence.zip) | 31.9MB | `8bb92c4ef881a06c8aa0f5036c035a6eaec5c246ed54891ae00670fb2f539f27` |
| [20261028-wm-sparse-magnitude-evidence.zip](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/20261028-wm-sparse-magnitude-evidence.zip) | 9.78MB | `86aaa94c10e11431b0ce7cefc70336b018ae2fac3aae48e3dae21b06f7b40009` |
| [20261028-wm-stream-overnight-evidence.zip](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/20261028-wm-stream-overnight-evidence.zip) | 19.59MB | `87a1a57905556eddf60cd788f421736c874c3be67b23deae193a66e3f7eaca83` |
| [AKD_MANAGED_IMAGE_MEMORY_1080P60.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_MANAGED_IMAGE_MEMORY_1080P60.mp4) | 9.14MB | `0e95acd3c8f82b6c632f73806c51d8d96ee205268f3c2819956937fb92b5ccc5` |
| [AKD_MANAGED_IMAGE_MEMORY_4K60.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_MANAGED_IMAGE_MEMORY_4K60.mp4) | 106.1MB | `c67d949dc78ee6ed7ecfe2829c6ff1cb1ff7aa69f778339f9156d14e9feed219` |
| [AKD_WORLDMODEL_MEMORY_4K60.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_WORLDMODEL_MEMORY_4K60.mp4) | 33.7MB | `a0e5269af9027c2f7dacbf8b48ef87fb2b4060b1ef94b8841f805b0467705a98` |
| [AKD_WORLD_MODEL_X.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_WORLD_MODEL_X.mp4) | 6.12MB | `4fe983296f71331beb7aaa0e4afe8658e4fb20340f38b8cf44ab8f3f0b8d1822` |
| [AKD_WM_MEMORY_DOWNLOAD.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_WM_MEMORY_DOWNLOAD.mp4) | 24.56MB | `a69db6bd9630a26d5cc497f0ec6bd96c716e4166bfd77c63df320a708d053e40` |
| [AKD_WM_NATIVE_MEMORY_4K60.mp4](https://github.com/MidasMulli/akd1500-lab/releases/download/research-20261028/AKD_WM_NATIVE_MEMORY_4K60.mp4) | 179.01MB | `96b431445d22de577ccc7918153952988b84346ffb0c2c94a58d83cf639ca6d7` |

Download with a browser or a standard HTTP client. Verify the SHA256 before extraction. Each evidence ZIP contains `PUBLIC_EVIDENCE_MANIFEST.json`; it gives every included file's byte size and hash. Paths are relative and must remain inside the extraction directory.

The silhouette bundle contains all matched native arms and captured RAM/output arrays. The sparse-retention bundle contains DEV/HELD streams, frozen encoders, CPU comparisons and all seven native cells. The stream bundle contains the five native cells, actual query/feedback journals, new-scene failure and independent audits. Repeated images are correlated, and selected simulator cohorts are not deployment samples.

Archive replay/audits do not prove a fresh hardware replication. Rebuilding a mapped programme needs the separately licensed SDK 2.19.3/runtime and its dependencies. Historical study launch wrappers are not reproduced as current permission. Frozen source and numeric inputs are kept with each experiment; run them from the documented study root. Some historical compiler helpers require path relocation. The public replay quickstart needs no SDK.

## Historical videos

The [through-October 25media release](https://github.com/MidasMulli/akd1500-lab/releases/tag/media-20261025) preserves40 byte-identical earlier MP4s and a SHA256 manifest. Their original Git commit remains available. Videos have moved out of the current source tree; Git history has not been rewritten, so full-history clones still include earlier objects.

## Software checks

```sh
python -m pip install -r requirements-replay.txt
python -m unittest discover -s tests -v
python -m unittest instruments.explorer.test_http instruments.explorer.test_tablet -v
```

Hardware controls remain off in the default server. The renderer is an instrument, not an electrical spike monitor. See [current status](STATUS.md) before interpreting a film.

The default test suite checks local numerical sources plus declared external media bindings. Release payload SHA256 digests were separately checked against GitHub at publication. Optional full RGB capture tests skip when their raw study directory is absent; the recorded RGB API rejection tests still run.
