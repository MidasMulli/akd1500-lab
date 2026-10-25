# CPU gate correction before compiler or physical admission

The first own-source CPU design admitted both the current and preceding key on
a query when they differed: current+previous+query > 1 did not require current.
It consequently returned ambiguous predictions. That rejected source and all
twenty seed/control screen results are preserved under failed_attempts/cpu01.

Correct the fixed manager to 2*current+previous+query > 2 (always-open variant
uses 2*current > 0). Current is now mandatory. This uses ordinary integer fixed
weights and a comparator, not a new silicon instruction. No device was opened
or programmed for the rejected candidate. Repeat the complete CPU screen before
compilation; outcomes, keys, seeds, stream order and endpoints stay unchanged.
