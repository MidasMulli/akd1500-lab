# Data, external tools and evidence boundaries

Dataset: MNIST, originally created by Yann LeCun, Corinna Cortes and Christopher
J. C. Burges. Downloads use the
[CVDF-maintained mirror](https://github.com/cvdfoundation/mnist), whose README
links the original training/test IDX files and acknowledges the original data
provider. The result manifest records exact URLs, MD5 identities, SHA256 values
and selected original indices. Original full compressed datasets stay local;
the public experiment archive contains the selected derived binary features,
labels, reference identities and owned measurement outputs.

Train: 60,000 originals; test: 10,000 originals. Exemplar selection is confined
to the first 10,000 training items. Development comes from the other 50,000
training items. The capacity revision's official test indices are disjoint
from every test image examined by its failed parent. Each test subset has 100
images per digit. Pixel threshold 128, centered 32×32 padding and 2×2 spatial OR
are fixed. Original grayscale values are not passed into the timing loops.

This is not BrainChip's provided MNIST model. Its
[official workflow](https://doc.brainchipinc.com/examples/general/plot_0_global_workflow.html)
uses a different trained/quantized neural network and includes simulator and
version-specific power-measurement stages. Do not borrow its accuracy or power
figures for this binary-exemplar benchmark or the installed AKD1500 card.

Compiler: installed Akida 2.19.3, virtual device mapping only. The preserved
error is a single-layer filter ceiling. No compiler internals, physical maximum,
arbitrary firmware execution or successful 3200-filter native program is implied.
Generated vendor binaries/runtime source are not redistributed.

CPU: Nano AArch64/core2, g++ 13.3.0, three repetitions of every selected test image
for every implementation and representation. The public source and recorded
source/binary hashes identify the measured program. Wall timers include query
packing/pooling/decode and timer overhead, with fixed bank setup outside the
hot path. Power, whole-host scheduling/IRQ work and continuous frequency were
not measured. CPU timings cannot be compared with prior synthetic AKD figures
as a matched substrate experiment.
