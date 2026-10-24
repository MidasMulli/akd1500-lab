# Initial native-learning policy candidates stopped at their CPU gate

No hardware execution was admitted. All connection-update forecasts use the
previously verified binary native rule; they are CPU forecasts, not new physical
learning results. The policy controls which delayed feedback is taught and which
logical population may learn. It does not replace the internal update machinery.

Equal total resources:128 records,32 connections each,512 binary inputs. The stock
and error-gated policies use four native classes. The split policies use two
logical64-record populations and eight teacher groups paired into four semantic
classes; this is one NPU, not two physical NPUs. Prediction has no domain label.

Development seed98401:

| Policy | Adaptation teaching jobs | Final old probe | Final new probe |
|---|---:|---:|---:|
|Stock|256|108/128|126/128|
|Error/confidence gate|83|118/128|121/128|
|Split stock|256|118/128|125/128|
|Protected split + gate|54|102/128|123/128|

The error gate saved jobs and improved retention here but lost5 new-probe correct
answers against a maximum allowed loss of2. The protected split fell below the
absolute retention gate and lost16 old answers relative to its matched-layout
control. A physically frozen old population does not guarantee whole-model
retention: an adaptive population can win the final readout with a wrong class.

Other initialization forecasts were mixed; neither candidate was consistently
qualified. No tuning within this frozen study and no physical seed expansion.
The original design source is preserved with its prospective hash in evidence.
`CPU_FORECAST.json` retains every prediction, teaching decision and failed gate.
