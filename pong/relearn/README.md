# Relearning, retention and CPU differentiation

The first changing-physics Pong pilot failed its prediction gate. This follow-up
separates a native learning-history problem from task learnability, then measures
the cost of replaying that exact task on the AKD1500 and CPU.

## Native three-arm diagnosis

All arms start with the same random connection initialization. They train court B
first. Two then train the original court-A rule identically; the third receives
protected A queries without the old A labels. All arms see the same subsequent
observations and query order. Fresh and relearning receive exactly the same new
A labels; frozen receives protected labels. **No B teaching occurs afterward.**
These are sequential runs of one 160-record learning bank on one chip, not three
simultaneously operating NPUs.

| Final endpoint, seed 85301 | Old A, frozen | New-rule-only A | Old A, relearn |
|---|---:|---:|---:|
| New A on separate development trajectories | 12/64 | 46/64 | 28/64 |
| New A on repeated teaching examples | 14/64 | 63/64 | 54/64 |
| B retained without further reminders | 47/64 | 47/64 | 47/64 |

The practical CPU baseline is a per-court FIFO of 64 labeled binary observations,
with nearest-neighbour prediction and newest-first distance ties. It receives
the same encoded inputs and permitted labels. It reaches 44/64 on the new A
development trajectories and 50/64 on B. It is a separate task algorithm, not
the CPU reconstruction of the AKD's binary learning rule. The two-example gap
between new-rule-only native and the CPU does not establish an accuracy advantage.

The original-rule and frozen native prefixes have identical complete RAM and
scores. The new-rule-only arm differs in its complete learning history, so its
18-example advantage over relearning does not uniquely identify stale visible
weights or hidden selector state as the cause. B uses a disjoint input-code
region; its preservation is evidence for this separated context, not arbitrary
overlapping-memory robustness.

The relearning arm fails the preset 65% new-trajectory accuracy gate. A bounded
one-factor CPU screen changes only connection count, to 32 or 48, across three
fixed development seeds. Neither candidate qualifies. Those are **CPU forecasts**,
not additional native runs. Confirmation and paddle control remain closed.

## Matched cost study

The separate transport study replays the recorded 2,240-forward relearning stream
using the existing `offload_bench` and `tools/run_cell.py`. Program bytes, input
bytes, labels, output scores and final learning RAM must match the original
sequential native capture. The exact CPU replay implements the same neural rule.
It includes input packing, output checksums and repeated fresh-state setup.

Four configurations—serial MMIO, DMA read/write, queued DMA and queued DMA with
a 25-µs wait—each run three times in a prospectively fixed, interleaved order.
Reviewing their input/submit/fetch timings motivated a separately frozen,
targeted **read-only DMA** follow-up. That follow-up is not presented as an
originally preregistered primary configuration. It also has three repeats each
for serial read DMA and queued read DMA with waiting.

All tested native paths cost more elapsed time and calling-thread CPU than exact
CPU replay on this small recorded task. DMA and waiting reduce caller CPU versus
our MMIO path; that is a transport improvement, not a neuromorphic CPU advantage.
For queued modes elapsed/input is batch-amortized cost, not single-event latency.
The queue has eight jobs and nine slots. Since its labels were already recorded,
this does not demonstrate batching fresh, causally dependent game feedback.

Native timing excludes programming, observation generation, reference checking
and full RAM capture. Calling-thread CPU excludes system/IRQ work. No direct
electrical energy, full-chip saturation or silicon-only timing is measured.
Native transaction time combines input transport, submission, completion and
output fetch; it does not isolate NPU computation.

## Public evidence and tools

- `results/2026-10-23-pong-relearning.{json,npz}`: all 6,752 native diagnostic
  forwards, every RAM snapshot, native packet headers, CPU task predictions,
  the frozen protocol and explicitly separated CPU forecasts.
- `results/2026-10-23-pong-substrate.{json,npz}`: twelve exact transport replays,
  26,880 native forwards, every score and complete final RAM.
- `results/2026-10-23-pong-read-dma.{json,npz}`: six targeted read-only DMA replays,
  13,440 native forwards, every score and complete final RAM.
- `visual/pong-relearning-4k.png`, `visual/pong-substrate-4k.png`: measured figures.

```bash
python -m unittest tests.test_pong_relearn -v
python -m pong.relearn.audit_public
# Optional scientific figure regeneration:
python -m pip install matplotlib
python tools/visualize_pong_findings.py
```

Hardware owner: `pong/relearn/serve.py`; same source/fixture/binary freezes,
global device lock, session/forward caps, cutoffs and owned-process cleanup as
the earlier Pong owner. `src/pong_relearn.cpp` adds same-host timing and the
separate C++ FIFO CPU learner. Every C++ CPU prediction is checked against its
Python implementation. `pong/relearn/model.py` is the stateful exact-rule
screen, validated against every score and RAM snapshot of the new native runs.

`pong/substrate.py` reuses the existing benchmark for the orthogonal cost study,
checking its freeze and original sequence before admitting another cell. Both
DMA directions pass a scratch-transfer smoke first. No installed driver or
vendor runtime is changed. `pong/substrate_archive.py` exports its evidence.

The learning study and both cost studies are closed. New execution requires a
fresh bounded envelope and prospective admission; passing transport correctness
does not reopen the failed prediction gate.
