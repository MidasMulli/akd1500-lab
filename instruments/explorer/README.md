# Measured AKD1500 Network Explorer

Orbit and select records, inspect actual stored connections and before/after RAM, step through recorded edits and compare matched controls. The default runs entirely from captured data and needs no chip or SDK.

```sh
python -m pip install -r requirements-replay.txt
python -m instruments.explorer.server --bind 127.0.0.1 --port 8892
```

Open http://127.0.0.1:8892/. Do not open web/index.html directly; it uses the local data API. Native controls are disabled.

## Included recorded profiles

| File under generated/ | Meaning |
|---|---|
| replay.json | Qualified three-bank learned-policy experiment,268 measured events,32 plastic records plus protected helpers |
| 20261026-pending-eight.json | Four-bank delayed-credit recording; version2tablet projection only, no four-bank live actor |
| 20261026-visual-memory-v4.json | Recorded RGB teaching/recall with actual images; its initial-recall quality gate failed |
| 20261027-repeated-native-memory.json | Repeated native image-descriptor replacement recording; separate from pixel-valued silhouette recall |

Select an alternate profile with `--replay instruments/explorer/generated/FILE.json` and an unused local port. Every profile is paired with a data-hash receipt. The three-bank live graph must never be substituted for another recorded profile.

## Inspect

- Orbit/zoom, select a record, inspect raw words, connection IDs, changes, learning thresholds and swap budgets.
- Switch measured controls, scrub events, follow edits, or inspect stored memory/readout panels.
- Adjust background opacity or nodes-only presentation; export a true-alpha PNG.
- The optional Electron shell provides a floating transparent desktop window. See [desktop](desktop/README.md).

Response bits/native scores and RAM edits are captured. Intermediate overlaps are derived; geometry, camera motion and timing are schematic. Connections are actual stored masks, not decorative links. A high score is not a calibrated probability. Full Chromium/WebGL execution on Tab5 remains unqualified; its native tablet receiver is separate.

## Live-worker source

The bounded live worker is included as research source. It requires separately supplied hardware/runtime and a fresh local qualification receipt. No machine's admission receipt or SSH credentials are distributed. The public checkout fails closed with native controls off; reading or replaying data never starts the worker.

If preparing a separately qualified setup, configure AKD_SSH_HOST, AKD_REMOTE_ROOT and AKD_REMOTE_PYTHON on your own machine. These must identify your intended private Nano; the browser does not supply remote command text. Hardware configuration is not part of the replay quickstart.

[Findings and limits](../../docs/STATUS.md) · [numeric evidence](../../docs/REPRODUCING.md)
