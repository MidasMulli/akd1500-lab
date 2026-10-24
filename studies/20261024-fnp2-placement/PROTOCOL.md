# True host-DDR record placement — prospective isolation

The standard Python SDK reproduces the same wrong-record update. Local driver
source at commit0847678 documents a separate reserved host-DDR mapping beginning
at0xC0000000; the live stock SDK advertises16MB. This is distinct from card SRAM
at0x20000000. Source identity is not claimed for the loaded driver binary.

Use the documented HardwareDevice.program_external API to point unmodified
program data at an owned2MB host-DDR mapping, with data starting64KiB into that
mapping. First a zero-forward CPU-copy/readback gate. Then the identical
8192-record below/above target fixtures: frozen below, learning below, frozen
above, learning above. Two forwards per session, every potential and complete
RAM compared to the frozen CPU reference. Read only the known FNP2 base register
at0xFCC00010 to verify actual placement; validate API record RAM against the
same owned host mapping before/after each forward. Our host writes stop before
programming; no native record patch or CPU teaching fallback.

This test changes data placement through an existing SDK API, not installed
firmware/driver or unknown register fields. Only mapped, kernel-accepted host
memory within2MB and the known read-only register aperture are accessible.
Native input/output control memory remains managed by the stock runtime.
All file descriptors, mapped buffers and program-info bytes outlive execution.

One zero-forward gate plus four sessions/eight forwards maximum. Every gate
must pass before the next. A mismatch stops and closes this study, preserving
partial data; no silent retries. Shared research lock, fuser-free admission,
90-second child timeout and unprogram cleanup. Scientific cutoff18:00UTC,
hard stop18:05UTC on2026-10-24. Direct latest usergo; previous studies closed.

Claim boundary: determine whether verified host-DDR placement changes the
update-address failure. No CPU advantage, silicon root cause, calibrated energy
or broad capacity qualification follows from this small test.
