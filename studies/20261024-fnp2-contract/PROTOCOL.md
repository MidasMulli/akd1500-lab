# FNP2 external-record qualification

Continue the user's direct hardware-mapping instruction and subsequent "go".
This fresh campaign does not reopen the closed RGB/transport campaigns.

The existing runtime places inaccessible caller program tracks into its bounded
device allocator. Here that allocator is the known card-system-SRAM aperture.
"External FNP2" is not assumed to mean host DDR. Read the documented runtime
FNP2 address register and require the complete record buffer to remain within
that owned SRAM region. No new driver/firmware/clock/address-space configuration.

Compiler-only: decode the 12-byte FNP2 track descriptor and exact 20-word numeric
record layout at 512 inputs / 544 records, 32 selected connections, two classes.
Check every initial weight bit against the seeded matrix. Two synthetic zero
words adapt to the established FNP3-style CPU selector; they are not native
storage. Whether that selector matches FNP2 learning is an explicit hypothesis.

Seeds 107101 weights / 107102 inputs. 64 fixed binary inputs, densities cycle
0/512/64/256/64/256/128/256, labels alternate. Same optimizer parameters and
initial buffer for all arms. Before hardware, compiled CPU frozen and learning
streams each run five repeats; independently replay scores and all state in
Python. Native stage one: eight frozen inputs, complete addressed/API readbacks.
Only after exact inference: eight teaching inputs. Any score or state mismatch
halts physical expansion; preserve partial outputs and RAM. Only after exact
learning: 64 frozen and 64 learning inputs, per-step full record snapshots.

At most four physical sessions / 144 forwards; each process 120-second cap.
Scientific cutoff 2026-10-24 16:00 UTC, hard stop 16:05 UTC. Exclusive research
lock, require device free, terminate only owned jobs. CPU and native transaction
timing excludes setup and diagnostic state reads. No global-capacity, host-DDR,
hidden-state-restoration or energy claim. No application/task-quality endpoint.
