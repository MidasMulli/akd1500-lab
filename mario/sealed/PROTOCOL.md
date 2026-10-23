# Sealed Mario visual-consequence study 2

New study; the first pilot stays sealed. Five fixed native initializations,
82901..82905. One FNP3 bank,512 inputs,192 records,32 connections per record,
three64-record groups: unsafe, safe, protected query. Same native binary
selection rule and parameters as the pilot. Query label2 has threshold32 and
must leave every captured RAM word unchanged. No restored hidden selector.

Freeze the manifest of encounter identities before collecting outcomes. Levels
1-1,2-1,3-1,4-1; reset seed82800. Initial no-button wait frames: training0/16/32/48,
development4/20/36/52, test8/24/40/56. Walk-right prefixes: training72/88/104/120/136,
development80/96/112/128, test76/84/92/100/108/116/124/132. This yields80 training,
64 development and128 prospective test contexts. No resampling. A prefix that
already died is retained as invalid, never silently replaced. Report counts by
level. These are held-out timing combinations on known levels, not unseen levels.

Two supplied90-frame motor primitives, inherited unchanged: jump immediately
for20 frames, or wait16 then jump20; hold right throughout. Success is alive
and advance at least70 pixels from decision start at the fixed horizon. Game
info is evaluation-only; acting features use current/8-frames-earlier RGB.

Predeclare two fixed encoders: legacy occupancy from the pilot; sparse RGB
projection over an8x8 area-resized crop y96..207 of each frame, subtracting reset
sky RGB, concatenating384 values, multiplying a fixed +/-1 matrix(seed82800),
and selecting the32 largest of256 projections. Place features in the action's
half of512 inputs. Both use192 records and balanced action-local random initial
connections. This is authored perception, not learned visual representation.

Collect only training/development first. CPU exact-kernel calibration uses
seeds82801..82803, four fixed shuffled epochs. Select encoder by mean development
selected-action success, tie legacy; no other candidates, epochs, thresholds or
ranking changes. Ranking remains max(safe)-max(unsafe), tie action0. Admit the
larger test only if the selected encoder has no opposite-label aliases, beats
frozen and the best constant in at least two calibration seeds and improves mean
development success. Otherwise stop after mechanism/live diagnostics, leaving
test outcomes uncollected. Freeze the selected pipeline/source hashes before
collecting test outcomes or compiling test fixtures.

Stages: (1) emulator repeatability, leakage/alias and CPU calibration; (2) virtual
mapping/score proof and one32-input mixed physical smoke, full per-step recount;
(3) first full native learned/frozen pair and recount; (4) only after inspection,
remaining fixed seeds and action-only/shuffled-feedback controls. All arms share
initial RAM and input/query order; frozen uses query labels throughout. Shuffled
feedback is one fixed permutation of training labels(seed82999), never test
labels. Count failures. No held-out tuning or silent retries.

Report all valid test contexts and separately the predeclared conditional subset
where exactly one primitive succeeds. Adequacy: at least80 valid test contexts,
24 conditional contexts, both successful-action identities represented at least8
times, and at least3 levels represented. If inadequate, retain descriptive first
pair only and do not expand. Consistent-learning gate: learned beats frozen in
at least4/5 initializations, conditional aggregate gain >=10 percentage points
over frozen, best constant, action-only and shuffled feedback; all-valid success
must not drop below frozen. Lower95% bootstrap bound of the learned-minus-frozen
conditional difference must exceed zero. Resample level/wait blocks, average
initializations within a context first; do not treat seeds as independent worlds.
Use10000 fixed bootstrap draws(seed83000). Report both bounds and raw counts.

Fresh episodes must execute native-selected actions and check actual outcomes
and acting-input hashes. Synchronised live demonstration uses the resident
physical bank, not precomputed CPU responses. Query/response events carry frame
indices, input hashes, scores, RAM hashes and wall times. During each fixed macro,
protected shadow queries may sample at10Hz for display only; they do not change
the already committed macro. Label these separately from action-selection queries.
Show measured match responses, not invented electrical spikes. Report achieved
rate, inference/transport latency and pauses. No frame-by-frame policy claim.

Local envelope:120 minutes, maximum24 physical sessions,150000 forwards,
180 seconds per process; no new launch within two minutes of expiry. Common
device lock and occupied-device refusal. Stop on source drift, mismatches,
query RAM edits, ownership, deadline or failed cleanup. Routine build/transport
repairs preserve attempts and require a prospectively frozen revision. No full
Mario competence, general RL, CPU/energy advantage or power-loss persistence claim.
