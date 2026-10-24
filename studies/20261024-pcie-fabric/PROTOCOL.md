# PCIe resident neural-fabric environment

Fresh campaign authorized by the October 24 direct user request. Prior studies stay closed.

## Frozen staging

1. Compiler and independent NumPy integer references, no physical discovery.
2. Two compiled, preallocated NEON CPU implementations (dense and input-event
   skipping), all pixels through both convolutions and exact binary learning.
   Verify both against independently derived spatial features, scores and RAM.
3. One eight-image native smoke, small cell, all scores and every RAM boundary.
4. After inspection, 96 images small cell, learning and frozen, serial first;
   nine-slot queue carrying eight images only after corresponding serial pass.
5. After inspection, first convolution width 64 then 128, same gates. At most
   three temporal repetitions per fixed cell. No automatic full-ring retry.

The acting program receives 32x32 RGB bytes. HRC first convolution, one CNP
spatial layer and the FNP learner remain in one compiler-generated resident pass.
Host does no feature extraction for the native arm. Two stride-two 3x3 layers,
8-bit first weights and 4-bit inner weights, binary activations, 512 learner
inputs, 128 records, 32 selected connections per record, two teacher groups.
All weights are frozen seeded characterization fixtures, not a trained vision
model. The host supplies class labels. No recognition-quality claim is tested.

Seed 106101 for weights and 106102 for 96 images. Same inputs and initial RAM
for frozen/learning/CPU arms. First 16 samples are warmup for reported timing;
CPU has five fresh-state repeats. No source or fixture changes after freeze.

## Endpoints and boundaries

Require every native potential and complete learner RAM to equal independent
CPU execution. Native serial snapshots each image; queued snapshots each eight
images. Preserve all attempts and stop expansion on failure. Timing includes
RGB transfer, enqueue, wait and output fetch. CPU timing includes feature
processing, axon packing and learning. Diagnostic RAM reads and verification
are outside those intervals, but affect the surrounding instrumented stream.
Programming and preallocation excluded and reported as an exclusion. No claim
of a production performance maximum, independent simultaneous teachers, direct
GPU DMA, complete fabric saturation, energy advantage or autonomous teaching.

Use existing stack, known apertures and research lock. Never alter clocks,
firmware, driver, PCIe configuration or unknown registers. Stop only owned jobs.
24-session / 4096-forward total cap, individual process 120-second cap. Scientific
cutoff 2026-10-24 16:00 UTC, cleanup deadline 16:05 UTC; close earlier when done.
