# Next source target: a native pointer held in learned connections

Assessed design only. Not implemented, compiler-qualified or admitted to hardware. It does not amend the closed R/N study or the zero-physical limit of this proposal. The source checks established a replacement primitive with a supplied class label; that is not the required native selector.

## Intended concrete milestone

Replace any of 32 old records without retrieving its old cue or having the host choose its destination. Let a native controller retain a five-bit ring pointer, gate the fixed teacher-class scan, confirm the new answer, and advance only after success. Protect the other 31 records and demonstrate multiple full wraps. This would be a designed allocator, not an emergent policy or an exclusive CPU capability.

The host would supply new encoded observations, true outcome labels and a fixed clock schedule. Content teacher labels would always scan 0 through 31. The host must not decode a pointer, supply a target mask, select the next class from a result, patch learned weights, or resend captured feedback. Readbacks remain audit-only.

## Avoid confusing a device buffer with on-chip learned memory

The current raw feedback buffers have a device-visible address. Their physical memory domain has not been independently established here. Store the durable pointer in five explicitly allocated learning-neuron records instead, and use raw buffers only for transient native feedback. Establish the buffer domain before making stronger physical-location claims.

Proposed source learner: eight teaching classes. Classes 0/1 retain the learned error/equality rule. Classes 2 through 6 hold one mutable bit each; class 7 is reserved. Each pointer record would have 30 common anchors, a bit-record identity anchor and a value bit, for 32 connections. A read cue activates all five identities and the value-1 bit: each selected neuron's measured firing bit would reveal its stored value. A static per-bit class scan writes value 0 or 1 through native connection changes. A changed value has overlap 31; a qualified learning threshold of 30 should allow a one-connection swap. Inactive helper records must remain ineligible for learning.

This is a hypothesis. It requires direct source and physical qualification of class partitioning, threshold behavior, selector coupling and all protected RAM. The comparison and pointer partitions share the same learning engine; no independence claim follows from disjoint RAM.

## Make room without using unqualified NP types

The existing comparison operand repeats each candidate/truth feature eight times, with 16 common anchors. An assessed compact alternative uses 27 common anchors, two candidate bits, two truth bits and a canonical marker: 32 total. Different pairs would overlap at most 30. Alternating canonical/swap bootstrap inputs could commit each comparator at learning threshold 30. This would require a different source competition configuration; the current source threshold-cap behavior must not be silently inherited.

A compact source input can carry one copy of each paired semantic/program feature and expand it through fixed layers. Assess 896 input bytes and 512 source units: the proposed learning-RAM stride is 4 + 896/32 = 32 words, hence 16,386 words including its two-word packet header. This sits at a suspected packet-size boundary and must be checked by the actual SDK/compiler. It is not proof that a single physical NPU supports the configuration. Reject if it needs additional unqualified placements or exceeds the bank limits.

Prior mixed FNP/CNP attempts already failed in the closed 20261027-native-memory-fabric study. Do not repeat those candidates or assert that the next graph frees FNP3 capacity. The existing qualified circuit uses seven FNP3s.

## Native transaction flow

1. Read the five pointer neurons without teaching and carry the measured bits into transient native feedback fields.
2. For each fixed content class 0 through 31, compare the static class code with the native pointer in the fixed neural layers. Gate retirement and new-cue learning only for the matching class. Keep the pointer unchanged for the entire scan so one transaction cannot overwrite all classes as the pointer advances.
3. Accumulate success only from the selected class's post-write recall, using its actual returned output. An ordinary prior recall must not count as write success.
4. At the fixed end-of-scan clock, compute the conditional increment with neural Boolean layers, including the wrap 31 to 0. A static per-bit format/teach sequence then writes the resulting five bits into their learning records. Keep the comparison neurons and all other pointer records protected.
5. On the next transaction, read the actual pointer records again. A stale software mirror is not sufficient evidence of persistence.

The source needs room for static controls, six transient bits (five-pointer cache plus transaction-success), the return header and full content readout. One assessed layout uses input words 218–223 for the six transient bits, an earlier return header/48-word readout, and a host prefix write that leaves those six words untouched. Exact offsets, masks and source/output geometry must be frozen and checked before execution; this is not an approved wire contract.

Three neural layers may suffice for the counter: first compute carry conjunctions, then the two XOR terms per bit, then OR them into the new bit. Duplicate pointer inputs or an explicitly qualified signed-weight precision are needed for the Boolean terms. No compiler or hardware support should be assumed from this algebra alone.

## Staged admission, not another application demo

First prove the compact comparator over all nine pairs and both marker phases. Then prove every pointer state and the 31-to-0 wrap, including blocked permission, failed recall and duplicate transaction controls. Require exact outputs, both full RAMs and protected helper state against independent scalar/C++ controls. Qualify the complete source/compiler graph before considering any physical prefix.

Only then freeze a separate prospective native study: a small prefix, matched frozen-pointer/no-permission controls, two full replacement wraps, and delayed recall of the current working set. Give it its own caps and stop gates inside valid operator authority. Do not reuse the failed natural-scene stream as confirmation. Approximate recall and new-scene usefulness remain separate unsolved requirements.
