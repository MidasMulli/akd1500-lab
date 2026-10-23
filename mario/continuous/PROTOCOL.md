# Continuous Mario native-decision pilot

Prospective controller-interface experiment; all older failed studies stay sealed.
One unchanged projection+observed-frame encoder, 1024 inputs, 192 records, 32
connections, 64 unsafe/64 safe/64 protected query records, same native learning
rule and safe-minus-unsafe ranking, ties action0. No learned perception claim.

At every 24 game frames query both candidates natively and choose max margin.
A0 holds right+jump for8 frames then right16; A1 holds right24. Execute the
chosen24 frames, observe actual survival and displacement, then teach that
same earlier decision input: safe iff alive and displacement>=12 pixels.
Terminate on death, environment termination/truncation, flag or1200 controlled
frames. No emulator reset/teleport, search, game RAM, future labels or hand-picked
escape during a controlled episode. Reset only between labeled episodes.
Outcome position/death information is teaching/evaluation only, never acting
input. Each arm must take at least two native decisions in a surviving episode.

Train: four known levels1-1/2-1/3-1/4-1, waits0/16, prefix lengths56/80/104/128,
prefix policies walk, jump8 every48 frames from phase0, same from phase24.
Preserve invalid prefixes. Evaluate both24-frame actions at exactly the same
rendered state using fresh prefix replay. One fixed encoder, no candidates or
outcome-driven tuning. Four class-balanced epochs, seeds84101/84102/84103.
Train first; no development/test prequeries. Frozen receives the identical
training inputs with protected labels and suppresses all online updates.

Development starts: each known level, wait4, walk prefix76. CPU reference loop
first, preserving results regardless of benefit; no tuning. Physical32 mixed
smoke and independent full-state recount, then seed84101 matched full/frozen
continuous development runs with synchronized captured RGB/responses/RAM.
Stop all expansion on mechanics mismatch, source drift, unsafe ownership,
cleanup failure, or deadline. Stop statistical expansion if first pair's total
controlled displacement is below frozen or learned survives fewer episodes.
A negative first pair still permits its matched diagnostic capture, not new seeds.

Only if the first pair passes those gates: freeze sources and open a distinct
confirmation start bank: each known level, wait8, walk prefix84. First pair,
then remaining two seeds after each raw review if no regression. Report total
displacement, alive-at-cap/flag counts, deaths, decisions, native updates and
learned-minus-frozen by start and seed. CPU constant-action0/1 loops are separate
model-free baselines. No reliable-learning qualification from this small pilot;
three seeds share four starts, known task families and one chip.

Captures use actual native query values at decision boundaries while the game
runs at target60 frames/s; game holds during native queries/feedback are timed
and preserved. No invented electrical spikes, shadow-policy substitution or
future replies on past frames. Full independent emulator replay must reproduce
all captured RGB frames, decisions, raw inputs, labels, outcomes and RAM.

New bounded100-minute envelope, <=20 physical sessions/150000 forwards, <=180s
per native owner and <=4096 forwards per stream. Device lock and owner refusal;
stop only owned children. No firmware/runtime/driver installation changes.
