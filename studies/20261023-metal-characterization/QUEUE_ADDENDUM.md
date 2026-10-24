# Prospective queue/pipeline discrimination

Authored after width and CNP screen inspection, before these two physical runs.
The accepted queue size of nine during learning contradicts the interface's
one-input description; it does not prove overlapping learning. Test actual
eight-input teaching bursts against the exact sequential learning rule.

Two arms start from the same compiled program and full native reset/program,
with the same 64 half-active binary inputs and alternating labels. Each burst
preloads eight inputs, withholds the queue producer publication, then releases
it once. Capture all 64 native output vectors and complete RAM after each burst.

* OFF: the runtime's usual learning mode, output-buffer-end selection.
* ON: change only the known pipeline/end-selection field through the runtime's
  existing helper; verify the actual register and unchanged other bits. Restore
  the normal setting and unprogram afterward.

The device may reject, stall or produce wrong results. Preserve the first
failure and do not expand that treatment. Timeout two seconds per completion,
180 seconds total process, existing owner lock and study caps. No clock,
firmware, installed-driver or undocumented-address modifications.

Changing buffer-end selection changes the counter's measurement boundary.
Therefore no ON/OFF counter ratio will be called neural speedup. An exact pass
establishes only bounded correctness of these teaching bursts under that control;
it does not establish simultaneous arithmetic, general concurrent labels or
the maximum hardware capacity.
