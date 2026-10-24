# Recorded native connection replay

The `neuron_visual` renderer turns the existing, independently checked window
archive into a 24-second schematic 3D graph. It performs no new hardware work.

The film shows eight consecutive controlled teaching updates from initialization
107101, with matched frozen hardware recordings. The wider source study tested
24 selected single-connection updates across three initializations. This is a
new view of those measurements, not a new scientific result.

Each of the 512 input points is an actual configured binary input channel. The
eight larger nodes are sampled learning records, identified by their global
record IDs. Every input-to-record edge comes from a captured connection mask.
Node brightness and size encode the returned integer match response. A record
outside the current class view is labeled `not queried`; a response is not
inferred for it. Full read-only recall then supplies all eight sampled scores.

The teaching response, captured RAM change and later read-only recall are shown
as separate presentation phases. This does not establish their internal timing
within a hardware execution. The first event is expanded to ten seconds;
following events get two seconds each. Camera movement and coordinates are
schematic. No electrical spike timing or neuron-to-neuron links are invented.

The native side changes exactly one connection pair in the selected record.
The frozen side keeps its original mask and returns 31/32 matches to the same
cue, while the native learned record returns 32/32. These are connection-match
counts, not classification percentages or a useful-task success rate.

The revised right panel shows the exact same 512-bit input and the eight
sampled response values from the matched frozen hardware run. Those responses
change as the cues change, even though the captured connection masks do not.
Its selected-row recall marker uses the same zero-to-32 scale for the native
and frozen values; the one-count difference is not visually exaggerated. A dash
marks a record not queried in the current class view, rather than an assumed
zero response. The left graph is unchanged.

## Reproduce

```bash
python -m coprocessor.neuron_visual --video --out /absolute/output/directory
```

Requires NumPy, Pillow and FFmpeg. The current production encoder uses macOS
VideoToolbox for the 4K master and libx264 for the 1080p review copy. This does not
require the Nano or AKD1500 to be connected.

The renderer first runs the complete public-archive audit, verifies each shown
connection mask and change against the saved full-bank states, and verifies the
frozen masks. It writes `recorded-events.json` and `SOURCE_BINDING.json` beside
the videos and preview frames. Existing failure evidence and the original film
are preserved.

Scope remains one FNP2 learning layer, host-scheduled teaching labels/class views,
and records in external Nano DDR. The original direct path remains failed. This
film does not qualify arbitrary continuous learning, autonomous multi-NPU work,
physical neuron positions, or an advantage over a CPU in latency or energy.
