# Record-stride isolation with fewer neurons

Prior reference:512-input/8192-record bank has80-byte records; correct target6500
and wrong target6600 reproduce via the stock SDK. A zero-forward probe confirms
default SDK records at0xC0000040 in actual host DDR, while the earlier custom
transport used card SRAM. Domain alone therefore does not remove the failure.

This fresh diagnostic reduces population to4096 and widens each record to1024
binary inputs (144bytes). Seed107201,32selected connections,2classes and the
unchanged learning parameters. CPU-only compiler/virtual validation and exact
Python record updates precede admission. Target3600 has byte offset518400;
target3700 has byte offset532800. These bracket the prior524288-byte separation
with different neuron indices. Each target is the unique highest eligible
class1record, score31, input0zero and input1one-selected-bit replaced.

Use standard SDK HardwareDevice API/default transport/allocator, with complete
record RAM and independent read-only host-DDR views at the registered address.
Run below frozen/learning then above frozen/learning; two forwards each.
Scores and every word remain strict endpoints. On mismatch, preserve the prefix
and close; no reference edits or application admission. This is an isolated
lower-population diagnosis, not renewed capacity expansion.

Four sessions/eight forwards cap. Exclusive research lock, fuser-free admission,
90second child timeout, unprogram cleanup and external handle-release check.
Source/fixtures/SDKbinary hashes frozen prospectively. No driver/firmware,
clocks/unknown registers changed. Cutoff18:00UTC/hardstop18:05UTC2026-10-24.
Direct latest usergo. Do not infer a silicon address width or root cause from
one stride comparison. Preserve every failed or invalid preceding attempt.
