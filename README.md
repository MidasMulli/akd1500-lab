# AKD1500 Lab

Independent experiments and instruments for **native learning, connected memory and model correction on BrainChip's AKD1500**. Recorded chip responses and connection changes are inspectable alongside matched controls.

The current result is bounded: native memory can learn and retain useful corrections for selected recurring situations while a larger world model stays fixed. New-scene usefulness and a fully native replacement manager remain open. Strong adaptive CPU controls also pass; no speed or energy advantage is established.

## Start here

| Explore | What you get |
|---|---|
| [Interactive Network Explorer](instruments/explorer/README.md) | Orbit, inspect stored connections, scrub measured events and compare controls; recorded replay needs no chip |
| [World-model memory replay](demos/world-model-memory/README.md) | New 32-record stream study, measured responses/edits and reconstructed simulator scenes |
| [Image-memory replay](demos/image-memory/README.md) | Four stored silhouettes, partial-cue recall and native source-gated revision |
| [Current findings and limits](docs/STATUS.md) | Native results, failed gates, controls and the next unbuilt mechanisms |
| [Evidence and downloads](docs/REPRODUCING.md) | Versioned numeric bundles, exact hashes and recorded MP4s |
| [Research index and earlier work](docs/RESEARCH_INDEX.md) | The preserved benchmark and experiment record, including negative results |

## Run the Explorer without hardware

A macOS/Linux host with Python 3.12 or later, NumPy and Pillow:

```sh
git clone --filter=blob:none https://github.com/MidasMulli/akd1500-lab.git
cd akd1500-lab
python -m pip install -r requirements-replay.txt
python -m instruments.explorer.server --bind 127.0.0.1 --port 8892
```

Open **http://127.0.0.1:8892/**. The default is recorded replay with hardware controls disabled. Click a record, step to a connection edit, switch controls, or inspect exact RAM before and after learning. The viewer also supports transparent and nodes-only modes.

For the static image/world-model replays:

```sh
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/demos/world-model-memory/` or `http://127.0.0.1:8000/demos/image-memory/`.

## Latest measured results

| Study | Result | Boundary |
|---|---|---|
| Native image memory | Primary final recall131/132; held-mask confirmation131/132; misleading updates leave content RAM unchanged | Four preassigned12 × 8 silhouettes; same objects and initialization; source-quality and pixel teaching supplied externally |
| First world-model retention | Eight associations across four situations/two settings;64/64 repeated return queries correct | Cached selected contexts, exact descriptor recall; native trained-on-both frozen control also passes |
| Causal Nano GPU stream | Native selected memory16/16 on the held second encounter versus2/16 for the world model;14 errors corrected | Selected recurring simulator contexts; host encoding/clocks/forecast mixing; held controls are software references |
| New-scene screen | **Failed:**124 distinct keys,0 memory hits, no forecast improvement | No held/native expansion after failure |
| Replacement feasibility |128 reference replacements protect the other31 records; independent C++ circuit checks pass | **Software/compiler only:** supplied class labels choose victims; native allocator is unbuilt |

[Detailed status and ancestry](docs/STATUS.md) include all actual native controls, original-physics checks, class imbalance, capacity limits and preserved failures.

## What the visuals mean

- Returned response bits and stored connection changes come from captured chip output and RAM.
- Input overlaps/intermediate activity are explicitly derived from actual inputs and captured state.
- Geometry, glow, camera movement and replay timing are schematic. These are not electrical spike traces or a physical die layout.
- World-model scene images are deterministic reconstructions matched to the recorded queries. The model's decoder artifacts remain visible. The AKD stores compact descriptors/outcomes, not these images.

## Hardware development

The bounded live worker and compiler/driver source are included for inspection. A new device requires its separately supplied BrainChip SDK/runtime, local transport configuration and explicit qualification. The public checkout does not ship a machine's hardware-admission receipt or provide a public hardware-control endpoint. No chip is needed for the default replay or software tests.

```sh
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python -m unittest instruments.explorer.test_http instruments.explorer.test_tablet -v
```

See [reproduction](docs/REPRODUCING.md), [Explorer reference](instruments/explorer/README.md), [roadmap](ROADMAP.md), and [MIT license](LICENSE). Third-party rendering components retain their own notices.
