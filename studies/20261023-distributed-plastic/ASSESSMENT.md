# Single-layer native-population mapping probe

Compiler-only, Akida 2.19.3, `MapMode.Minimal` with an explicit FNP3-only policy.
No physical device was opened and no native capacity or performance was measured.
The original seven-cell probe ran inline; its full mapping/rejection receipts
are retained in the public handoff-contract JSON under `single_layer_compiler_probe`.

| Input features | Neuron records | Compiler selection under this policy |
|---:|---:|---|
|128|512|One FNP3|
|128|1,024|One FNP3|
|128|2,048|FNP2 requested; rejected|
|128|3,584|FNP2 requested; rejected|
|512|512|One FNP3|
|512|1,024|FNP2 requested; rejected|
|512|2,048|FNP2 requested; rejected|

The single layer does not distribute across several FNP3s in these tested cells.
The imposed policy rejects external-memory FNP2. Neither observation proves a
silicon maximum, lack of another mapping mode, or impossibility of a supported
distributed population. The subsequent six-NPU experiment instead uses three
separate plastic banks connected by fixed FNP encoders. Its successful native
result must not be attributed to this compiler-only survey.
