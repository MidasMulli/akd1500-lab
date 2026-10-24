# Resident multi-NPU learning study

Read [the assessment](../studies/20261024-resident-spatial/ASSESSMENT.md) for measured results, CPU controls, compiler failures and claim boundaries.

```sh
python -m coprocessor.resident_archive audit
```

Portable replay needs only NumPy; compiler preparation requires the user's Akida2.19.3. Hardware compilation uses build_resident.sh and separately supplied matching vendor runtime/headers. No vendor SDK source, generated program binary or runtime library is redistributed. resident_run.py is the exact closed-study runner, not a new launch authorization.

```sh
python -m coprocessor.resident_visual --out /absolute/output --video
```

The recorded replay needs NumPy, Pillow and FFmpeg (macOS VideoToolbox for the4K master). It audits the archive before rendering. Returned FNP3 potentials and connection bits are measured; features are independently calculated; routing diagram, geometry, camera and expanded playback timing are schematic. The frozen128-neuron grid changes with actual returned responses while all frozen weights remain unchanged.
