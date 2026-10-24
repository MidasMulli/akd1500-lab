# Standard SDK transport isolation — prospective October 24

Direct user instruction: investigate the off-target FNP2 update and progress
only after inspecting each stage. Previous studies remain closed.

Use byte-identical 8192-record programs, initial RAM and below/above target
inputs from the previous boundary study. First independently replay both
frozen and learning prefixes on CPU. Then use the installed public Python
HardwareDevice API and its default device driver, without the LabDriver or its
write wrapper: frozen below, learning below, frozen above, learning above.
Two inputs per session. Capture each response and complete API learning RAM.
Unique winner and expected RAM are fixed before hardware access.

A pre-forward API failure counts as a failed attempt. A wrong state stops that
session and closes this study; no retries or relaxed checks. No performance or
root-cause claim follows solely from reproducing a failure in another path.
No firmware, installed driver, clocks, SDK files or unknown registers changed.
Exclusive research lock, device-free admission, child timeout 90 seconds,
unprogram cleanup and fuser verification required. At most four physical
sessions/eight forwards, including zero-forward failed admissions. A fresh
protocol is required for independent subsequent diagnostic hypotheses.

Scientific cutoff 2026-10-24 18:00 UTC; hard stop 18:05 UTC. Research aim:
separate our transport adapter from the stock SDK path, not yet attribute the
failure to silicon or claim whole-chip limits. No application expansion here.
