# Native signal-gated predictive memory

The learner stays enabled with constant class 0. Prediction probes stay below the native learning threshold; outcome inputs cross it. A fixed second FNP3 provides the readout. Three seeds qualify learning, selective adaptation, query protection and retention with exact full-SRAM replay. The CPU remains faster, and the paced T202 co-running comparison shows no practical AKD resource advantage.

Read the [assessment](../studies/20261024-predictive-feedback/ASSESSMENT.md), [prospective protocol](../studies/20261024-predictive-feedback/PROTOCOL.md) and [GPU addendum](../studies/20261024-predictive-feedback/GPU_ADDENDUM.md).

```sh
python -m feedback.archive audit
python -m feedback.corun_audit audit
python -m feedback.visual --out /absolute/output --video
```

Portable audits require NumPy only. Visuals add Pillow/FFmpeg and use the existing neuron renderer helpers. Hardware preparation requires the separately supplied Akida 2.19.3 SDK and matching runtime; compiler preparation never discovers a physical device. Build scripts accept private runtime/header paths. No vendor source, library or generated program is included. The saved study is closed; run.py and corun.py reproduce its admissions, not new execution authority.

The synthetic encoder fixes context identity and feedback anchors; only the outcome association is learned. A host supplies observations and bounded queue descriptors. This is not indefinite recurrence, learned context encoding, self-generated prediction errors, a power-loss durability claim, electrical spike trace or CPU/energy advantage.
