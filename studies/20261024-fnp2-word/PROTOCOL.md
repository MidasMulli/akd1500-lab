# Same winner; connection-word location intervention

The1024-input/4096-record diagnostic reproduces the same512KiB metadata
separation as the512-input/8192-record program. This fresh lower-risk probe
keeps the exact1024-input program and initial RAM, and selects record3640 in
both conditions. Its metadata word131043 is below131072, while connection
word131072 is exactly at that word-offset boundary.

Two sealed inputs each remove one selected bit and add one previously absent
bit in the SAME connection word. Both produce a unique class1winner score31.
Low-word condition edits input word0, expected record word131044. High-word
condition edits input word28, expected record word131072. CPU-only decode and
full-state replay must pass before admission. No alias model is substituted
for the correct learning reference.

Standard SDK/default allocator, actual registered host-DDR address, read-only
mapped full-record checks, source/fixture freezes, no host record repair.
Order low frozen, low learning, high frozen, high learning; two forwards per
session. Any state mismatch stops that session and closes the study, retaining
all partial states. Four sessions/eight forwards maximum; exclusive lock,
fuser-free admission,90second child timeout and parent cleanup verification.
No extra forward occurs after a corrupt state. No installeddriver/firmware,
clocks or unknown registers changed. Latest direct usergo; cutoff18:00UTC,
hardstop18:05UTC2026-10-24. Earlier studies remain closed.

Endpoint: Does failure depend on which connection word is edited, even with
the same selected record and correct metadata position? This does not prove
hardware address width, a universal limit or a silicon erratum.
