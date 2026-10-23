# Resident burst instrument, prospective gate

At most three additional physical sessions within the original 24-session cap.
First 16 samples of n128/u8, then 64 samples of n512/u512, then a 256-sample
confirmation only after exact scores and full learning RAM pass.

Eight input buffers and eight descriptors are prepared before queue publication.
The queue write pointer is withheld while descriptors are filled, then published
once. Poll the known dense-input completion counter for the final descriptor,
assert zero host writes during this interval, then drain every output and compare
all scores and full final learning RAM to the sequential CPU model.

Report setup, queue-release-to-observed-completion and output-drain separately.
The middle interval includes PCIe publication and observation, excludes setup
and output readback, and is **not core-only silicon latency**. It cannot be
compared to complete CPU result latency to claim an application speedup. The
full resident transaction includes all three phases. Busy completion polling
remains host work. Raw output correctness is required for every queued sample.
