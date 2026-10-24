# Native fabric, memory and routing: evidence map

October 24 follow-up: [native class windows](WINDOWS.md) qualify selected high-address
learning with records in verified host DDR and controls in card SRAM. This uses
one FNP2 with host scheduling. The original direct-learning boundary is preserved;
its cause and unrestricted learning address coverage remain unresolved.

## Documented purpose

The PCIe variant is intended to work beside an application processor, run
networks in the fabric, and support models needing host memory. M.2 provides
integration/power/thermal convenience. It does not imply a discrete GPU-sized
processor or a guaranteed latency advantage for small outputs.

The reviewed PCIe datasheet V1.5 describes a separate image convolution engine,
32 NPUs in eight four-NPU nodes, packet routing, local neural storage and
1 MB dual-port system SRAM for data transfers. Its NPU-storage description is
nominal; usable budgets must be distinguished from raw physical capacity.
See [reviewed sources and versions](DOCUMENTS.json).

## Observed and executable

| Layer | Evidence | What remains unresolved |
|---|---|---|
| Physical link | Sysfs: vendor 1e7c/device a500, 5.0 GT/s, two lanes; installed `akida-pcie` binding | Effective link utilization, power-state transitions; loaded driver/source identity not fully established |
| PCIe apertures | Three 4 MB BARs observed; original harness uses known 1 MB register/system-RAM windows; window follow-up also verifies a separate kernel-backed host-DDR mapping | BAR aperture size is not SRAM capacity; original loaded driver/source identity remains incomplete |
| Input converter | Compiler HRC endpoint (0,0,0); actual 32x32 RGB to binary features agrees with independent integer math | Larger image/channel/stride/precision throughput surfaces |
| Neural positions | SDK 2.19.3 target: 32 CNP1-capable positions; 21 also CNP2, seven also FNP3, one also FNP2 | Types overlap at one position. Descriptor support does not demonstrate every mode/position or simultaneous execution |
| Position organization | Eight four-slot nodes; reserved DMA endpoints (1,1,0) and (1,1,1); HRC is separate | This is logical addressing, not a measured die floorplan |
| Usable SRAM description | SDK per-NPU descriptor: 44,800 weight bytes and 43,008 input bytes | Allocator budgets are not direct measurements of total SRAM or bank/port concurrency |
| Learning records | Tested RGB learner: 128 records, 512 binary inputs, 32 selected connections; complete 10,248-byte packet readback | Records are state serviced by engines, not 128 separately observed physical arithmetic units |
| Native communication | HRC -> CNP1 (1,2,1) -> FNP3 (3,3,0) passes every score/RAM check | Not arbitrary routing, measured router hops, all directions, independent packet sources or cross-chip links |
| Learning execution | Scores and each captured boundary match an independently reconstructed binary update rule | Hidden selector is reproduced behaviorally; it is not directly read out or proven restorable by visible RAM alone |
| Queue/pipeline | Eight images in nine allocated slots execute exactly; installed runtime turns pipeline off when learning is enabled | A runtime policy is not proof the silicon cannot pipeline learning |
| Host involvement | Host supplies images/teachers; intermediate spatial features remain native | This is not autonomous teaching, persistent flash learning or direct GPU-to-card transfer |
| DMA isolation | Zero-forward byte checks at 128/512/3072 bytes each way; four transport profiles retain exact learning state | DMA caller savings do not measure kernel-wide CPU or energy; DMA writes increase elapsed cost here |

## Compiler format frontier, not a silicon maximum

The subsequent [FNP2 study](FNP2.md) resolves the decoder boundary below and
finds a separate off-target learning update with records in card SRAM. The
later [class-window study](WINDOWS.md) reproduces the failure through the stock
SDK, verifies actual host-DDR placement, and qualifies a guarded workaround
for selected single-swap teaching probes. The original path remains failed.

An additional compiler-only enumeration preserves all format changes. At
512 binary inputs, 512 records produces one decoded FNP3 packet (40,968 bytes).
At 544/560/576/1024 records, this compiler selects an FNP2 external-memory record
format that this harness deliberately refuses to execute. At 1024 inputs,
288 records remains FNP3 (41,480 bytes), whereas 320 selects FNP2. At 2048
inputs, 128 records remains FNP3 (34,824 bytes).

These are observations of specific compiler choices and our decoder boundary.
They do not establish the maximum neurons, largest model or total usable chip
memory. In particular, the 512 -> 544 switch cannot be attributed solely to the
44,800-byte budget: layout/rounding and alternative native modes also matter.
The previously qualified FNP3 capacity and whole-chip fixed-graph results remain
separate from this compiler-only extension.

## Highest-impact remaining gates

1. **FNP2 update-address contract.** Stock SDK, different record strides and
   adjacent-record interventions narrow a record-base-related failure; its
   root cause remains unresolved. Native class windows work for selected
   unique single-swap probes. Test hidden selector behavior under arbitrary
   window switches before expanding learning claims. Host-DDR ownership and
   provenance now pass bounded checks; unrestricted address probes are unnecessary.
2. **Production output and transport.** Qualify compact binary/class outputs
   from the same RGB network. Include output encoding/decoding on both CPU and
   AKD sides; keep all native learning/state endpoints unchanged.
3. **Native sharing and overlap.** Separately vary image converter work, CNP
   work, number of independently taught banks and competing ingress. Confirm
   solo/mixed exact state first, then measure common-timeline service. A routed
   same-job result is not simultaneous execution or independent teacher support.
4. **Electrical boundary.** M.2 lacks the PCIe evaluation board's current
   monitor (board guide V1.0). Existing power tools are a bench design, not an
   assembled/calibrated rig. Interpose/calibrate complete card power before
   publishing energy ratios; measure host and card consistently.

No new application is needed to answer these questions. We now have a tested
RGB-to-native-learning co-processor path on which to isolate them.
