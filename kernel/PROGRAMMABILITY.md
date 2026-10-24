# Programmability: exercised interfaces and unresolved internals

Read-only audit of 170 C++ headers/source files in the installed external Akida
runtime. The [public inventory](../results/2026-10-23-programmability.json)
contains file hashes and symbol locations, not redistributed vendor code. A
keyword search is a discovery aid; the conclusions below also use the reviewed
programming, learning-toggle and DMA entrypoints. This is a scoped source audit,
not exhaustive silicon reverse engineering.

| Layer | Current evidence | Boundary |
|---|---|---|
| Host runtime/transport | User-authored bounded MMIO/DMA adapter, queue publication and timing instruments execute against known interfaces | Driver/runtime optimization does not replace the NPU learning mechanism |
| Native network | Compiler-generated weights, thresholds, records and routing run as resident native graphs | Supported-operation composition, not demonstrated arbitrary instruction execution |
| Learning configuration | Optimizer parameters, learning register tracks, enable/disable and learning-memory APIs are exposed | Parameters/memory access do not prove upload of a new update algorithm |
| Native update algorithm | Existing binary rule reconstructed against physical outputs and learning RAM in earlier studies | New policies in this campaign failed CPU qualification; no physical algorithm improvement |
| Arbitrary learning instruction loader | Not established in the API/source reviewed | Neither accessibility nor physical impossibility is proven |
| Fixed circuit versus internal microcode | Unresolved by these sources | Do not label the learning engine definitively hardwired or reprogrammable |

`DeviceProgrammer::program_single_pass` traverses records and applies register
tracks, RAM records and EPG data. `configure_learning_mode_single_pass` selects
the learning/inference register track. `HardwareDeviceImpl::toggle_learn`
changes that configuration and pipeline behavior. These are concrete controls;
none is a demonstrated arbitrary learning-instruction upload interface.

The only `firmware` keyword found in the audited runtime was an AKD1000 memory
map comment. The AKD1500 SPI flash file's `instruction` and `opcode` occurrences
are flash-command terminology. These hits do not demonstrate a programmable
AKD1500 neuron update processor. No flash, firmware, clock or installed driver
was modified in this campaign.

Official [Engine documentation](https://github.com/Brainchip-Inc/akida_examples/blob/main/doc/user_guide/engine.rst)
describes a C++ host/MCU library that programs and communicates with the chip.
Official [AkidaTag firmware](https://github.com/Brainchip-Inc/AkidaTag) targets
an external Nordic nRF5340 MCU paired with the AKD1500 over SPI. Replacing that
MCU firmware is different from replacing the AKD's internal learning algorithm.
The [optimizer API](https://doc.brainchipinc.com/api_reference/akida_apis.html)
and [edge-learning parameter example](https://doc.brainchipinc.com/examples/edge/plot_2_edge_learning_parameters.html)
document configurable aspects of the built-in rule.

The productive remaining paths are controlled native graph construction,
configuration/packet semantics, and useful workload/energy tests. A proposed
internal algorithm replacement first needs an identified execution interface
and a bounded observable proof, not a firmware keyword or an unqualified RAM
write. Visible RAM restore also does not establish complete resumed learning
state; replay the full learning prefix when that state is required.

The [October 24 native-capability study](../capability/README.md) adds physical
evidence for matched fixed operations, causal contribution of all 11 large-graph
compute parameter regions, and 32 queued teaching events while a host worker
blocks. It also separates the runtime's measured 128-slot allocation ceiling
from a silicon bound and retains the failed 128-event stage. None establishes
an arbitrary instruction loader or a replacement native update algorithm.
