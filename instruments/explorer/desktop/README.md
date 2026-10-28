# Optional desktop shell

The Electron shell presents the local recorded Explorer as an always-on-top transparent, nodes-only window. It connects to http://127.0.0.1:8892/; start the read-only Python server first. This shell does not add chip access.

Run `pnpm install --frozen-lockfile` in this directory, then `pnpm start`. The lockfile pins the dependencies. Read package.json before installing; Electron is a separate dependency. The web viewer works without Electron. Click-through mode has a restore shortcut; keep that shortcut available. The renderer and controls are the same recorded instrument, not a new experiment.
