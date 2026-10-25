# Initial RAM assertion correction before physical launch

Attempt04 compiled the requested FNP3 layers but failed an own-source assertion
that incorrectly expected zero compiler auxiliary words. The activation threshold,
plasticity constants and connection count are nonzero record auxiliaries. Compare
the initialized learning threshold/swap count and every binary connection against
own seeded values; preserve all compiler auxiliaries and headers for subsequent
bit-exact captures. This corrects the preflight verifier, not hardware learning.
The failed receipt is retained; no physical program or forward occurred.
