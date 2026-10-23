# Recognition repairs routing while downstream weights remain fixed

**Native modular-repair gates passed in all three initializations.** Two resident
learning banks have different semantic jobs and separately enabled teachers.
A fixed native comparator/encoder connects them; a native reducer supplies the
four final destination support counts. All five stages map to FNP3 NPUs in one
hardware pass. No host-derived intermediate identity is injected.

| Endpoint, each seed | Recognition repair | Frozen | Routing-only repair |
|---|---:|---:|---:|
| Old signatures after shared teaching | 64/64 | 64/64 | 64/64 |
| New signatures before repair | 16/64 | 16/64 | 16/64 |
| New signatures after repair | **64/64** | 16/64 | 16/64 |
| Old signatures after repair | **64/64** | 64/64 | 48/64 |

Seeds 97101/2/3 are initial connection seeds on one chip and one synthetic
four-beacon world. The new probe noise is distinct from repair teaching noise.
Recognition receives identity labels; routing receives destination labels.
Their labels differ, and only the selected bank learns on a teaching event.
Neither label nor domain/identity/destination metadata enters sensor features.

In recognition repair, routing's complete captured RAM remains unchanged while
upstream changes recover the original semantic message and correct destination.
Old signatures retain 100% performance without refresh teaching after the
change. The routing-only control fails to recover changed signatures and loses
one old signature class. This localizes a useful repair intervention in this
task; it does not establish general self-diagnosis or autonomous teacher choice.

The instrument returns exact native outputs and per-step RAM, not electrical
spike timing. All 9,280 native forwards and 18,560 complete bank snapshots match
the C++ and independent Python reconstructions. The three arms have identical
initial programs, inputs, native prefix outputs and prefix bank states before
treatment changes. Fixed encoders/reducer read back unchanged. Ten sessions pass;
there are no failed physical sessions. Earlier compiler/build setup failures
remain in `evidence/REPAIRS.json` and local complete logs.

A post hoc CPU comparator with two exemplars per identity and a learned routing
table also reaches 64/64 on changed and retained probes. Functional correctness
alone is not a substrate advantage. No electrical energy was measured; diagnostic
transactions include mode/transport costs and RAM verification is separate from
the native forward interval. There is no whole-chip saturation, native search,
recurrent loop, simultaneous independent stream, hidden-state restore or
power-loss persistence claim.

The study is closed. Native jobs were released; live checks found the device
and shared owner lock free. Sources and numerical evidence are packaged in the
public lab. The visual is a source-bound, time-expanded logical replay with
reconstructed responses and measured RAM edits/final outputs.
