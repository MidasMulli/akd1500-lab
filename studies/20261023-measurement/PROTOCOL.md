# Hardware measurement and capacity frontier

User directs measurement before further controller features, whole-chip investigation,
custom power instrumentation, visual evidence, and a public tooling repository.
The preceding active-recovery source remains held, uncompiled and unlaunched.

## Admission and accounting

Maximum 24 physical programming sessions, 150,000 forwards, 180 seconds per
owned native process. Scientific cutoff 2026-10-23 14:25 UTC; cleanup by 14:30.
Only the common device lock admits work. No frequency, voltage, driver, installed
SDK, firmware or unrelated deployment changes. Every compiler or execution
failure is retained. A changed source requires a prospective repair receipt.

## Questions and controls

1. What do equal binary-association scores and learning updates cost on an
   optimized CPU bitset implementation versus the actual FNP3?
2. How do costs change with input width, configured neurons, selected connections
   and placement? A compiler success is not a physical capacity measurement.
3. How many different engines can participate correctly in one mapped workload,
   and does increasing resource use improve useful work per time?
4. Can card-level electrical energy be measured independently of host power?

Compiler frontier: (128 inputs,8 neurons), (512,64/256/512/1024),
(1024/2048/4096/8192,1024). Native admission is sequential: a 16-sample
small-cell smoke; 64-sample pilot; then passing larger cells with 64 samples.
Each complete output vector and final learning RAM must equal the frozen CPU
selector model. An unexplained mismatch stops that family before expansion.
Successful cells can receive three interleaved native/CPU timing pairs of 256
samples. CPU repetitions amortize timer overhead and are disclosed, with fresh
initial state each repetition. They are not independent device samples.

CPU and native implementations receive identical binary inputs and labels. The
CPU uses population counts and the independently reconstructed selector/update
algorithm, rather than a slow dense neural simulator. Native timings include
required input transfer, submission, completion polling and output readback.
They exclude programming, compilation, RAM readback, verifier calculations,
disk logging and the previous artificial 20 ms audit waits. Busy polling is
counted as host CPU work. This is a learning-kernel benchmark, not image-service
latency or total application energy.

Inference and learning are separate endpoints. Initial smoke must validate
learning; freeze/inference measurements require unchanged RAM. Subtracting
inference latency from learning latency does not isolate intrinsic silicon
update time, so no such causal claim is allowed.

Larger mapped/mesh families must have a recorded single-pass resource map,
independent expected outputs and fixed-RAM/learning checks before admission.
No claim of full-chip utilization from node count or compiler placement alone.

## Power

Physical inventory: PCIe AKD1500, SDK SocDriver unavailable; Nano INA3221 rails
are module domains. No direct card power measurement is currently connected.
Design an external shunt/INA228 recorder, explicit card supply boundary, raw
voltage/current/energy records and host phase markers. Hardware presence,
calibration, time synchronization, no-bypass current path and a known-load test
are required before any physical energy result. Synthetic recorder tests are
always labeled synthetic. Report gross card energy, powered-idle baseline,
incremental energy and matched host-plus-card boundaries separately.

## Public package and figures

Publish own source, tests, reproduction commands and sanitized compact results.
Exclude licensed runtime sources, binaries, credentials, private correspondence,
absolute personal paths and unreviewed historical records. Include every failed
cell and distinguish mapped, executed, correctness-passing and unmeasured.
Visual geometry is a logical resource map; no physical die or electrical-spike
claim. Quantitative charts use linear/log axes explicitly and preserve raw data.
