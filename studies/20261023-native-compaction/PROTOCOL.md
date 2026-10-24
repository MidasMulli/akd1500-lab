# Native graph compaction with matched external I/O

Direct operator "Go" on improving native execution and learning. This separately
bounded study tests a native graph rewrite; it does not replace the internal
learning engine. It is a synthetic representation co-design test, not a useful
application or CPU/energy superiority claim.

Freeze seed98501, widths256/512 and a512-bit external input consisting of four
identical128-bit copies. The direct graph uses512-input detector records whose
32 selected connections lie in the first copy. The rewritten graph first applies
a fixed on-chip four-copy AND/consensus encoder, then the same detector weights
in128-input records. Both graphs threshold detector scores at15 and produce the
same two count outputs. Inputs and external output contracts are identical. No
host compaction, intermediate host activation or hidden preprocessing is allowed.
The equivalence claim applies only to the declared four-copy domain.

Map the detector to FNP3(2,1,0) and readout to FNP3(3,3,0) in both versions. The
rewrite adds FNP3(1,3,0); count all its processing cost and memory. Reject other
kernel types/multipass mappings. Test latent densities0/64/128, three shuffled
blocks of16 each. Every output must equal an independent binary matrix reference.
Capture all declared fixed NPU memory regions before/after each session. Readback
headers identify requested targets, not hardware-discovered physical coordinates.

Compiler/virtual preflight first. Native smokes at width256, then width512,
direct before compacted. Smoke32 events plus48 repeated frozen suffix forwards.
Inspect exact outputs, complete memory regions, idle/config counter controls and
service distribution before full144-event runs. Only then admit two additional
temporal repeats per cell, alternating path order. Maximum16 sessions/4000 forwards,
180 seconds per process under the saved cutoff. Source/binary/fixture freeze,
exclusive owner lock and device-free check required. Stop/preserve any failed gate.

Efficiency gate: compacted median half-active DMA service and paired resident
burst monitor cost must improve by>=5% at a width in all three full sessions.
Report losing widths and host latency separately. An extra NPU, fixed encoder
memory and the input-domain restriction are explicit tradeoffs. Counter scope
is buffer service, not arithmetic-only cycles. No firmware/clock/driver changes.
