# Identical current symbols, native carried context

The same external symbol X produces four different predictions depending on the
preceding context A–D. That context arrives from the preceding AKD output in device
SRAM. After one initial zero seed, the host writes only the current-symbol half of
the next input buffer. It does not construct or copy the history half used for acting.
Two FNP3s hold eight records and a fixed state-code decoder.

Fresh external labels change two context-to-outcome associations repeatedly, without
reloading weights or resetting thresholds. These are eight changes between two seeded
synthetic rules, not arbitrary sequence learning or learned discovery of context codes.

| Five initializations, eight changes per resident run | Native | Frozen | Unlabelled | History erased at probes |
| --- | ---: | ---: | ---: | ---: |
| Changed-association checks | 240/240 | 120/240 | 120/240 | 0/240 |
| Unchanged-association checks | 240/240 | 240/240 | 240/240 | 0/240 |
| Ordinary context-code checks | 480/480 | 480/480 | 480/480 | 480/480 |
| Cumulative native connection replacements | 1,280 | 0 | 0 | 1,280 |

The full cues and incomplete cues after 1 ms / 20 ms dwell all pass. In the history-erased
control, the host deliberately clears only the carried half before X. Captured learning
RAM at every probe is identical to the intact arm; X then produces no response. This
qualifies explicit output-SRAM context. It is distinct from the earlier query-history
effect outside the captured learning bank.

A strong CPU state machine recognizes the same actual current symbols, retains a
context byte and updates a four-entry table from the same labels. It also passes
480/480 predictions. A latest-symbol-only ablation passes 120/480; that deliberately
memoryless ablation is not the headline CPU baseline. No CPU or energy advantage is
established. Repeated probes across five seeds are correlated, not independent trials.

## Replay and reproduction

```sh
python -m temporal.archive audit
python -m temporal.audit_mutations
python -m temporal.visual --out /absolute/output --video
python -m temporal.video_qa /absolute/output
```

Portable replay needs NumPy. Rendering additionally uses Pillow and FFmpeg. Virtual
compilation and physical reproduction require the separately obtained Akida 2.19.3 SDK
and matching runtime; headers, libraries and generated programs are not distributed.
`build.sh` takes `AKIDA_ENGINE_ROOT`, `AKIDA_RUNTIME_LIBRARY` and `FLATBUFFERS_INCLUDE`.
Physical admission uses a new private authority/state, exclusive device lock, source
freeze and staged CPU/native gates. The saved study is closed.

[Assessment](../studies/20261024-temporal-context/ASSESSMENT.md) ·
[Prospective protocol](../studies/20261024-temporal-context/PROTOCOL.md)

## Scope

Four externally supplied context symbols, one following query symbol, eight native
records, engineered initial weights and externally addressed destination classes.
Context representation and feedback pointers are designed, not discovered by learning.
Each current symbol needs host scheduling; this is not a self-scheduling chip or a
real physical closed-loop task. The host seeds one blank buffer and writes 128 current
bytes per event. Only the explicit ablation writes 128 history bytes at each probe.
Readbacks diagnose activity after each event; no no-instrumentation comparison was run.

The film replays returned codes and captured connection masks. Activity is inferred
through a fixed decoder. Sixteen output-channel links per code are aggregated in the
diagram. Layout, flow particles, camera and expanded playback timing are schematic;
no electrical spike timing, physical neuron coordinates or hidden register is measured.
Older FNP2 high-address and native multi-pass replay failures remain unresolved.
