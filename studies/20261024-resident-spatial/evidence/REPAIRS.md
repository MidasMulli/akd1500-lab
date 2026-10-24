# Preserved attempts and repairs

- Compiler attempt01: pooling directly into FullyConnected rejected. No physical device opened. Source and failed BUILD.json preserved.
- Compiler attempt02: 512-channel projection at four weight bits rejected as exceeding filter capacity. Smaller cells mapped, no physical device opened. Source and BUILD.json preserved.
- Compiler attempt03: prospectively encoded the same -1/0/+1 projection coefficients at two bits, with a following native projection convolution. All six fixture cells independently matched integer arithmetic and mapped as single-pass FNP3. This was frozen before any physical run.
- Numerical archive first invocation: import of the compiler helper required absent Akida on Mac. Replaced that dependency with standalone integer reference arithmetic; hardware sources and measurements unchanged.
- Visual preview first invocation: string/Path mismatch in provenance SHA helper after previews rendered. Corrected path normalization; full production render and source binding then completed. Measurements unchanged.

No failed physical session was observed. All18 physical admissions /1072 forwards passed saved-array checks. The first two compiler failures do not establish an intrinsic hardware maximum; they establish constraints of the tested native mapping and encoding.
