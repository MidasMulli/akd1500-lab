# Original larger Mario development: stopped at calibration

The task manifest and two RGB encoders were frozen before collecting training
and development outcomes. The [protocol](PROTOCOL.md) proposed 80 training,
64 development and 128 test contexts on four known level families, with two
fixed 90-frame actions. No original test outcome was collected and no physical
session was run for this parent study.

CPU calibration failed. Legacy learned scores were 55, 55, 55 out of 64 versus
frozen 55, 55, 51; projection learned scores were 51, 55, 51 versus frozen
48, 48, 48. The best constant scored 55/64. Each encoding had an opposite-label
alias: development level 4-1, prefix 128, waits 36 and 52, action 1. The rendered
playfield was pixel-identical; only the excluded HUD differed. More updates
cannot recover information absent from the acting input.

Status: **STOP_CALIBRATION_GATE**. Those are CPU-model results, not measured
native controls. No main-test expansion was admitted. The later clock revision
and separate confirmation do not turn this failed gate into a pass.

Provenance limitation: the fixture producer was not separately source-frozen
before this failed CPU-only screen. Original bank/collector source snapshots
were subsequently preserved and matched their pre-outcome design hashes;
generated fixtures and outputs remain saved. This is a real documentation gap,
not retrospective preregistration of that producer.
