# Native temporal context: prospective staged protocol

One resident two-FNP3 model; eight binary learning records and a fixed state decoder.
The first128 input/output channels encode prior native output; the last128 encode only
the current external symbol. After each completed native forward, the host writes only
the current-symbol half of that output buffer. The next descriptor reads the preceding
native output directly. Except in the declared severed-history control, the host does
not write, copy, decode or construct the history half used for acting. Readbacks are
diagnostics after the event. The host schedules every event and supplies external labels.

Four context symbols A–D produce four context codes. The same following symbol X
must produce four distinct predictions based on the retained context. Initial weights
encode a seeded prior; this is not learning from scratch. The changed stream swaps two
context-to-prediction associations. Native labels address the outcome record; no host
weight repair occurs. Two presentations per changed context. Query X is full16 bits,
or12 bits after1ms/20ms dwell. Context inputs have32 active current bits. The readout
outputs16 active history bits and zero current bits. Threshold24 and fixed SDK learning
competition0.75, initial plasticity0.5, minimum0.25, decay0.125. Eight alternating map
changes revisit two maps, not unseen-world discovery. Source/feature placements vary
with seeds112101,112201,112301,112401,112501.

Stages: virtual/CPU preflight; seed112101 native e0 (8 old probe events), then e1,
e2,e8 (8+32*epochs forwards); then frozen, unlabelled and severed-history e8 controls.
Only after raw output/full learning RAM and causal-input audits pass, admit each other
seed's four e8 arms. The severed arm receives native teaching but zeroes the history
half before post-teaching X probes; this is the explicit ablation, not normal operation.
Frozen disables native learning after the old prefix. Unlabelled keeps learning enabled
but receives no teacher labels. All controls receive identical current-symbol streams.

Require native recall for every changed and stable association at all three query
variants, unchanged context codes, one program load, zero weight restores/threshold
resets, exact scalar+packed CPU replay at every captured boundary, previous-output
pointer identities, zero normal history-half writes, full descriptor label verification,
and fixed decoder memory unchanged. A strong CPU state machine uses actual current
symbols and its previous context; it receives the same teaching and must also pass.
A symbol-only comparator is an ablation, not the headline CPU baseline. No latency,
energy, electrical spike-trace, hidden-register or CPU-superiority claim is admitted.

Failure stops dependent expansion. Preserve source, error, partial observations and
unused later stages. Routine source/transport repairs require recorded prospective
amendment and fresh suffix qualification. No alteration of a failed receipt.
Maximum32 physical admissions /9,000 forwarded events. Deadline/ownership in private
AUTHORITY.json. All physical runs require an exclusive research lock and free device;
unprogram only the model owned by this instrument. No other process may be stopped.
