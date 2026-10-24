# PCIe neural environment assessment

The user challenged the assumption that an application-processor host was a
poor fit for AKD1500. Manufacturer material explicitly supports PCIe offload
beside ARM/x86/RISC-V application processors and drop-in M.2 deployment. We
built that neural environment before making another application.

## Physically demonstrated

* Raw 32x32 RGB -> HRC 8-bit convolution -> CNP 4-bit spatial convolution ->
  512 binary features -> 128-record FNP3 learning, one resident compiler pass.
* First-layer widths 16/64/128, frozen and learning, serial then queued eight
  images in nine slots. Largest queued cell has three temporal repeats.
* Two independent C++ NEON CPU implementations and a separate Python integer/
  binary-learning replay agree on all scores and complete captured states.
* Primary: 17 physical sessions, 1,544 forwards, 704 full RAM snapshots;
  12 CPU runs with five fresh-state repeats, 5,760 CPU sample executions.
* Transport follow-up: 28 physical sessions (one zero-forward), 2,328 forwards,
  312 full RAM snapshots; all four transport profiles retained.

The combined public archive independently recounts 45 physical sessions,
3,872 forwards and 1,016 RAM snapshots. Captures are complete 10,248-byte learner
packets. Queued captures occur every eight images, not after every image.

## Measured limitations

The strongest tested CPU control remains faster overall. With learning on the
widest cell: CPU 248.4 µs/image, mapped queued AKD 446.8, read-DMA 440.0,
read/write-DMA 707.3. The latter reduces calling-thread CPU to 99.2 µs/image;
kernel workers/interrupts and complete energy are unmeasured. This is a
latency/host-activity tradeoff, not proof of whole-system efficiency.

The installed runtime disables its single-pass pipeline when learning is
enabled. This explains a configured difference between the two arms; it does
not establish an unavoidable hardware restriction. We did not change that
policy or any firmware/driver/clock setting.

SDK and compiler resource enumeration are kept separate from physical limits.
The target describes seven FNP3-capable positions and one FNP2-capable position,
with other roles overlapping. The compiler changes from FNP3 to the currently
unqualified FNP2 external-memory format as some single-bank cells grow. That
does not mean the chip cannot execute larger learners/models.

Two authoring attempts failed before physical execution: CNP kernel convention,
then spatial-to-FC ordering. Both are preserved. No physical run failed in this
campaign. Seeded kernels and synthetic images test execution; this study does
not test classification usefulness, task adaptation, unseen generalization or
the Nano world model. No new X performance claim is qualified.

## Next work

Continue metal/runtime characterization: qualify FNP2 host-memory semantics,
compact readout on this same graph, and independent stream/teacher interference.
Build and calibrate the external card-power boundary before electrical claims.
The detailed gates are in [the fabric map](../../coprocessor/FABRIC_MAP.md).

Both bounded campaigns closed normally before their cutoff. No owned physical
jobs remain. Public evidence is replayable without hardware or vendor SDK.
