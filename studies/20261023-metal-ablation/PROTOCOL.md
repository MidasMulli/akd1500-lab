# Causal per-NPU contribution and physical memory localization

Separately bounded follow-up to the completed execution characterization, under
the same direct user instruction to test core hardware questions. At most six
physical sessions and 1,024 forwards; 180 seconds/process. No old-study reopen.

Use the exact compiler-generated 256-channel CNP graph, two compute NPUs and one
fixed readout NPU. Initial binary weights, input stream, graph dimensions and
readout remain fixed. Four prospective treatments: baseline, raise A's 128
thresholds from 15 to 32, raise B's thresholds, raise both. Inputs have at most
32 connected bits/neuron, so threshold 32 suppresses the designated responses.

Before hardware: pass virtual exact outputs; require identical packet layouts
and that only the threshold memory region of the expected distinct NPU changes
in each treatment. Preserve decoder/format mistakes before corrections.

Baseline first. Read all five generated weight/threshold/readout memory regions
from their actual NPU targets before and after 64 frozen native forwards. Require
exact equality to the generated parameters. Inspect before admitting A, inspect
before B, inspect before both. All outputs must match an independent integer
reference; suppressing A must remove only output A's contribution, and vice versa.

Do not call a successful ablation physical clock overlap or energy advantage.
Do not mutate firmware, clocks, installed drivers or unrecognized addresses.
Use source/fixture/binary hashes, explicit admissions, the shared owner lock,
two-second completion bounds, saved failed receipts and owned-process cleanup.
