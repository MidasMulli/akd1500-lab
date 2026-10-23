# Prospectively admitted mesh integrity ladder

The successful compiler family uses 128 binary input channels in a 1x1 tensor,
1/7/15/31 fixed channel-permutation convolution layers, then a 64-neuron FNP3
association learner at (3,3,0). Every memory neuron has 32 selected connections.
The fixed permutation layers are composed for an independent CPU oracle.

The mapper assigns 27,29,31,32 distinct compute NPUs respectively. The largest
image has 31 CNP1 placements plus one FNP3 placement, one pass and no runtime
host computation between layers. This is configured-resource coverage, not a
measurement of simultaneous execution utilization, maximum throughput or useful
CNN accuracy. Learned communication/search is not claimed.

Hardware gates: 16 samples for the 27-NPU cell; then 64 samples for each
successive passing cell; then one exact 32-NPU repeat if all scores and complete
learning RAM match the CPU reference. Only the final FNP3 is plastic. All fixed
permutations, parameters, layer assignments and source hashes are retained.
Every native process must unprogram and release the shared lock before the next.

The first spatial 4x4 fixture was not admitted because the independent flatten
ordering failed. Initial mapping callback and API assumptions were repaired
before hardware; their failed logs remain in the private raw study. No failure
was promoted to a silicon limit. No end-to-end timing comparison uses precomputed
CPU mesh fixtures to claim an accelerator advantage.
