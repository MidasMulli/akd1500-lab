# Replacement feasibility addendum — software only

No card was opened. This is a separate proposal, not a reopening of the closed R/N studies or an admission to hardware.

## What passed

- The first Python reference rehearsal made 64 replacements per seed, using fixed seeds 27102811/12: 128/128 new answers recalled and all 31 other record states unchanged after each replacement. Both impaired controls recalled 0/128 new answers. It used 102,912 reference forwards.
- SDK 2.19.3 mapped the new 32-class, one-unit-per-class configuration on a virtual AKD1500. The actual mapping uses seven FNP3s; it does not demonstrate a mixed-FNP layout.
- A second, smaller circuit witness used two fixed victims (0 and 17), two consolidation protocols per write, and three arms. Python and independent C++ outputs and both full learning RAM arrays agreed on 1,072 / 1,048 / 1,048 forwards. Wildcard retirement recalled both new answers and protected the other 31 records; no-retire and permission-blocked controls recalled neither.

The 128-replacement reference rehearsal and the two-victim compiled-circuit witness are different checks. The C++ executable evaluates a numerical oracle; it does not execute the SDK-compiled graph on hardware.

## What has not passed

The target record is supplied by the synthetic schedule/class label. A native counter, target-mask generator and replacement policy have not been implemented or qualified. The first rehearsal visits all classes on fixed software clocks, using a software target mask. This is not yet native destination selection, nor self-managed memory. Global reference selector state (rot/cursor) changes even when other records' RAM is protected, so independent learning engines are not established.

The inherited C++ success line contains labels such as `native_mailbox_return:true` even in CPU mode. Those labels are not hardware evidence. The controlling status files say `physical:false`; the new executable rejects native and stream-native modes before opening a fixture or constructing a device.

Two failed builds are preserved: the first omitted the canonical visual_memory helper path; the second omitted the canonical src path for readout_bench.cpp. The final repair changed include paths only. Participant, cases, encoder and mapped programme stayed unchanged.

## Review and reproduction

`RESULT.json`, `COMPILER_CPP_STATUS.json`, all three fixtures, Python TRACE expectations, independent C++ outputs, failed/successful logs, programme bytes and exact owned dependencies are included. The separately licensed BrainChip SDK/runtime, static runtime library and linked executable are not redistributed.

Clean extraction verifies every file and all three full-array comparisons without the original repository or SDK. Rebuilding the virtual mapping needs SDK 2.19.3 plus NumPy and these paths on PYTHONPATH: source, owned_sources, owned_sources/tools. Rebuilding C++ needs the separately supplied runtime archive and SDK headers, plus owned_sources/visual_memory and owned_sources/src include paths. The historical driver scripts retain their original absolute paths; relocate those launch paths explicitly before running them elsewhere. The participant source/fixtures are unchanged.

No new scenes, encoder fits, physical sessions, speed/energy comparison or X publication. R generalization and capacity failures remain in the main study. This addendum does not qualify the next native experiment.

## Public source edition
The public check_source.py changes repository/source path discovery and writes REPLAY_RESULT.json, preserving the original RESULT.json for a clean checkout. Its original source hash and frozen numerical results remain in RESULT.json. No native allocator is implemented, and no hardware mode is admitted. Compiler launch wrappers are excluded because they were tied to private bounded study clocks and separately supplied runtime paths.

Run the software-only fixed-seed rehearsal from the repository root: `python proposals/20261028-blind-replacement-source/check_source.py`. This uses no SDK/device and does not implement native selection.

RESULT.json is the earlier reference-phase receipt. COMPILER_CPP_STATUS.json records the later SDK virtual-mapping and independent C++ witness checks. Neither is a native allocator result.
