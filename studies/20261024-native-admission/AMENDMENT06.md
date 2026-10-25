# CPU recency baseline and explicit mechanism gates

Before pressure admission, correct the bounded CPU dictionaries to update LRU
recency on reads as well as writes; the previous screen used write recency only.
This is an analysis-only comparator correction. It changes no native fixture,
input, weights, label, program or endpoint. Both CPU policies still have eight
entries and receive the same raw observations and outcomes. Preserve prior source
hashes and recompute comparator denominators from each captured stream.

Make the short physical mechanism gates explicit in the portable verifier:
native8 rejected/8 admitted with edits; open0 rejected/16 admitted with edits;
closed16 rejected/0 edits. All three existing captured runs already meet these
conditions. Read-only retrieval qualification is a separate prefix stage.
