# Small real-image bank failed the frozen quality floor

No physical access or native timing. This study is closed.

The direct and spatially compacted banks used the same 480 MNIST exemplars,
48 per digit. Development passed at 165/200 and 167/200. On its sealed balanced
1000-image test, raw reached **814/1000** and compacted **817/1000**, below the
frozen **850/1000** accuracy requirement. The three extra correct classifications
do not qualify an absolute-accuracy pass. No threshold, seed or exemplar change
was made within this study.

The separately named capacity revision preserved these exemplars, increased
coverage and used a different test subset that excluded these 1000 images. It
does not turn this failure into a pass. The original source, predictions,
identities, protocol, closeout and source hashes remain preserved.

See the public `results/2026-10-23-image-capacity.json/.npz` archive and
`python -m recognition.archive audit-public` for an independent packed-XOR
reconstruction of every decision.
