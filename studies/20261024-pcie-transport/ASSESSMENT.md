# Fixed-graph transport isolation

This prospectively frozen follow-up keeps the primary RGB program, weights,
images, teacher labels and learning endpoints fixed. One zero-forward copy
check validates both transfer directions at 128/512/3072 bytes. Separate
eight-image learning smokes pass for all three kernel-DMA profiles before the
full cases. Four profiles × frozen/learning × three temporal repeats all pass.

| Profile | Frozen elapsed / caller µs per image | Learning elapsed / caller µs per image |
|---|---:|---:|
| Mapped I/O | 290.3 / 289.9 | 446.8 / 446.1 |
| DMA read | 283.7 / 133.1 | 440.0 / 289.4 |
| DMA read+write | 707.4 / 99.1 | 707.3 / 99.2 |
| DMA write | 762.1 / 303.1 | 763.5 / 303.6 |

Medians of three run medians; first 16 images excluded. Full per-image and
eight-image timing samples are retained. The strongest primary CPU is about
248 µs/image. DMA both ways trades elapsed time for lower caller activity;
neither elapsed nor caller time alone is a calibrated energy measurement.

28 owned sessions, 2,328 forwards, 312 complete learning-state snapshots. All
scores and captured states independently match Python and both compiled CPU
references. No mathematical, model, workload, driver or clock substitutions.
No hardware arm failed; slow DMA-write arms remain in the public archive.
