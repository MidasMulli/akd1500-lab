# Clock revision: mechanism verified, usefulness gate failed

[Revision 1](REVISION1.md) was registered before physical execution or any
original test outcomes. It appended observed elapsed emulator frames and
balanced the training classes. Those two interventions were not causally
isolated. The alias disappeared, but the inherited development gate failed:
projection learned 55, 55, 51 out of 64 versus frozen 50, 46, 48 and constant
55 in every seed. Legacy learned 55, 53, 52 also failed. Projection was selected
by the fixed mean-development rule. The original 128 test contexts remain
unopened; none of these development values is a sealed-test result.

Six physical diagnostic cells reproduced the projection results exactly:
**161/192 learned, 144/192 frozen, 165/192 constant**. C++, native responses and
the independent Python selector/RAM reconstruction matched on every forward.
The [public archive](../../results/2026-10-23-mario-sealed.json) preserves the
six cells, two mixed smokes and both interactive captures: ten sessions and
11,614 native forwards. All closed cleanly and released the device.

Live full/frozen streams replayed the same training/development preparation,
then ran four selected development cases with actual chip queries while the
game advanced. Both succeeded in 2/4 cases. These were selected examples, not
an accuracy sample. Full feedback produced 45 connection replacements over
four outcomes; frozen RAM stayed fixed. Achieved full-arm rates were 56.03
captured frames/s and 8.41 query pairs/s. Actual RGB frames were independently
recreated, all encoded inputs were matched to their source frames, and every
native response/RAM snapshot was recounted. Wall-time pauses are retained in
the [4K60 film](../../visual/mario-live-responses-4k60.mp4).

Live preparation included protected development prequeries. Such queries can
advance the hidden selector despite unchanged visible RAM. The later
[confirmation](../20261023-mario-confirmation/ASSESSMENT.md) therefore uses
train first and no held-out prequeries. That is a separately registered study,
not a repair of these observed development results.

The CPU selector was extracted verbatim from the existing native benchmark
into a shared header, then checked against the prior Git source and independent
Python reconstruction. The interactive transport added streaming and readback;
it did not change the installed driver, firmware or learning algorithm. Native
outputs and real RAM edits are observed; electrical spike timing is not.
