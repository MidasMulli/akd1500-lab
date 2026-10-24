# Runtime queue ceiling measured; owned cleanup passes

One physical session, **zero neural forwards**. The known owned six-FNP program
is programmed, all three learning and three fixed parameter regions are read,
queue allocations are requested, all memory is re-read unchanged, then the
program is cleanly unprogrammed and the handle closed.

|Requested slots|Returned slots|
|---:|---:|
|1|1|
|9|9|
|33|33|
|65|65|
|128|128|
|129|128|

The installed 2.19.3 runtime source explicitly clamps single-pass capacity to 128.
This confirms a software/runtime allocation ceiling, not a silicon queue bound.
The earlier 128-event held-ring technique needed 129 slots and therefore failed
its guard. This diagnostic neither retries nor rescues that failed learning
stage. No smaller fallback learning arm or full-ring experiment was admitted.

Program storage remains alive through cleanup/destruction in this diagnostic.
A caller-buffer lifetime hazard was found in the earlier actor; the exact cause
of its SIGSEGV is not uniquely isolated. Its original cleanup remains unconfirmed;
this separate zero-forward owned cleanup is confirmed. All admissions closed,
no owned jobs, device released. Numeric observations are included in the primary
native-primitives public archive, with the reviewed runtime-source hash.
