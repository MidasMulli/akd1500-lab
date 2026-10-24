# Near-budget native record writes, guarded host transport

Fresh prospective follow-up after all four 1024/2048/4096/8192 FNP2 populations
pass independent inference, per-step learning and addressed state checks.

One additional population: 11,264 records × 512 binary inputs × 32 selected
connections. Same seeds/parameters/eight-event prefix. Expected record buffer
901,120 bytes. Runtime memory report retained; total allocation must remain
inside the known 1 MB card-system-SRAM window. This deliberately retains
headroom; it is not an allocator/silicon maximum claim.

Compiled CPU frozen/learning five repeats and Python replay before hardware.
Eight native frozen inputs, then only after independent pass eight teaching
inputs. Every output and complete record-state boundary must match CPU.

After initial programming/readback, interpose an owning HardwareDriver wrapper
which refuses any host write intersecting the actual registered record extent.
All experiment/runtime transport writes traverse the wrapper. Capture the
attempt counter and total guarded write calls. This observes our harness;
it is not hardware access control or proof about arbitrary kernel/device agents.
Require zero record-write attempts while native learning changes stored state.

2 physical sessions / 16 forwards; each process <=120 seconds, exclusive research
lock. No driver/firmware/clock/unknown-register changes or host-DDR mapping.
Scientific cutoff 2026-10-24 16:00 UTC, hard stop 16:05 UTC. Stop on any failure;
no retry or increased population. Preserve all partial data and stop only owned
jobs. Eight events prove bounded operation, not converged quality, long-horizon
retention, all records being trained, peak throughput or energy advantage.
