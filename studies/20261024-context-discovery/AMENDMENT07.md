# Held-out CPU receipt serialization repair before new physical suffix

The held-out CPU trace passed every array and endpoint assertion, but writing its
AUDIT.json failed because a summed NumPy boolean became a non-JSON int64. Preserve
cpu_s113101_unseen_c12 and its arrays/receipt as a failed audit. Convert the summed
booleans to Python int, refreeze, and rerun CPU under a new suffix. No native program,
fixture, learning/reference behavior or scientific endpoint changes. No physical
held-out session had been admitted at this failure.
