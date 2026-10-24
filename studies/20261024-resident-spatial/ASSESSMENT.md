# Resident spatial fabric and native local learning

The X draft was updated and saved, not published. A separate fresh study then qualified RGB input through multiple fixed convolution NPUs and a native local-SRAM FNP3 learner. All18 physical sessions /1072 forwards and36 CPU admissions passed independently replayed numerical checks. The full numerical archive is in results/20261024-resident-spatial.{json,npz}; run python -m coprocessor.resident_archive audit without the SDK or device.

## Demonstrated

At512 spatial filters, the mapped path is HRC image input -> two CNP spatial partitions -> one CNP projection -> one FNP3 learner, in one resident hardware pass. No registered host DDR weight memory is supplied. All fixed CNP configuration-memory packets read back exactly before and after; every full1026-word learner packet is captured after every event and matches three CPU implementations. The single-pass FNP3 runtime readback path uses an NPU SRAM configuration read, not a copy of the initial program. HRC configuration is not covered by this CNP readback routine.

Each64-event stream has8 initial queries,32 labelled teaching presentations,16 repeated queries and8 novel queries. Native connection weights and learning metadata change only during teaching. Frozen twins never change RAM. The24 query events after teaching preserve every learner word. All learned responses independently match a dot product of that event's features and its actual pre-event connection masks. Every removed connection was inactive in the teaching cue and every added connection was active;32 connections per record are conserved. Native changes occur inside the assigned teacher class.

At the largest width, three independent initializations produce24 actual connection-edit events and266 connection swaps in total (each swap removes one bit and adds one). Thirty-two teaching presentations do not imply32 connection changes. The film replays eight sampled teaching events and eight read-only queries from seed107501. It shows128 derived feature channels and eight actual learning records, plus all128 measured frozen response values. No electrical spike times or physical neuron coordinates are observed.

| Cell | Actual connection-edit events | Connections added (same number removed) | Native median µs |
|---|---:|---:|---:|
| c128_s107501 | 8 | 91 | 1268.17 |
| c256_s107501 | 8 | 89 | 2247.97 |
| c512_s107501 | 8 | 90 | 2264.10 |
| c512_s107601 | 8 | 87 | 2260.36 |
| c512_s107701 | 8 | 89 | 2256.28 |
| c8_s107501 | 9 | 109 | 354.48 |

## Resource result and limits

For seed107501 with learning, measured native medians are2247.98µs at256 filters and2264.10µs at512. Interleaved native repeats give2249.34/2247.30µs and2263.58/2264.86µs respectively. Nominal dense convolution MAC counts grow from4,972,544 to9,723,904; spatial mapping grows from one to two CNPs. Near-constant pipeline latency is a capacity-scaling observation, consistent with resource parallelism; it does not measure per-neuron simultaneity or identify the transistor-level cause. The weights and projection width change with each workload, so this is not an identical-program NPU-isolation intervention.

The fastest measured CPU control at512 filters is the four-core NEON/OpenMP path (~633µs/image). The fastest single-core path is ~2210µs; native is ~2264µs. No overall CPU latency advantage is established. CPU controls execute all convolutions, packing, matching and the full native connection-selection rule; they are substantial exact controls, not a claim of a globally optimal CPU implementation. Five CPU repeats versus one initial native run, native temporal repeats at256/512, no randomized CPU/native order and no locked clocks: performance evidence is characterization, not a universal benchmark.

Latency includes native raw input transfer, submission and output fetch. Per-step memory observers and reference verification are measured outside the timed path (~3ms at512 filters). Those diagnostics must be included in any complete live-monitoring service budget. The controller currently polls and consumes substantial CPU; quiet completion/interrupt or batching is an explicit next engineering gate. Process CPU time includes CPU-control worker threads but is not calibrated energy. No card power measurement or GPU co-run has been made in this study.

CPU caches matter: live sysfs reports core2 L1D64KiB, private L2256KiB and cluster L32MiB (shared cores0–3). CPU coefficient/threshold/learner arrays total158728bytes (~155KiB), excluding activation buffers, code and allocator/runtime state. This model may be substantially cache-resident. Local NPU SRAM therefore does not automatically prove a DRAM-bandwidth advantage. GPU contention requires a measured co-running comparison; the AKD host I/O/controller still uses Nano resources.

This is synthetic seeded feature characterization, not a trained useful vision task, temporal prediction result, generalization result or autonomous feedback loop. Only the FNP3 is plastic here; the three CNPs remain fixed. Volatile SRAM residency across queries is not persistence across power loss. Prior bounded host-sleeping execution does not prove indefinitely self-sustaining native feedback.

## Failures and disposition

Compiler attempt01 rejected max pooling directly into FullyConnected. Attempt02 rejected the512-input projection at4-bit weight encoding. Attempt03 uses the identical -1/0/+1 projection arithmetic encoded at2 bits, after a native projection convolution, and was frozen before physical access. Failed attempts and exact source copies remain preserved. Two nonphysical packaging/renderer defects were repaired; measurements unchanged. See evidence/REPAIRS.md.

The study is closed; no owned device job remains. The subsequent user instruction 'go build, and really drive innovation here' authorizes a separate prospective study of feedback autonomy and a useful predictive coprocessor/GPU co-running comparison. This closed study and its source freeze remain unchanged.
