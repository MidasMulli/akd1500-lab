# Native context-record allocation — October24,2026

**Qualified: the chip allocates and recalls distinct memory records without a
context-ID label or a host-selected record.** The representation of context remains
an engineered four-symbol history with a fixed third-order feature basis. This is
one component toward the planned loop, not a completed autonomous predictor.

## Measured result

| Endpoint | Native single-class learning | Trained then frozen | Always untrained |
|---|---:|---:|---:|
| Distinct histories recalled, five initializations |80/80|20/80|0/80|
| Original context-to-record IDs retained |20/20|20/20|Not learned|
| New-order recall, five initializations |640/640|Only first seed tested below|Not tested|

The matched first-seed held-out stream has128 windows. Native recall identifies
128/128 using the original record IDs. The trained-frozen control responds to25/128,
exactly the windows belonging to its four learned histories; it rejects all other
histories as unallocated. That is expected missing knowledge, not wrong known-context
answers. All five native held-out runs cover all16 histories in a new raw-bit order.
No teacher labels, weight reload, history reset or connection changes occur during
held-out recall. Five initializations share a fixed synthetic stream; these counts
are repeated windows, not640 independent environments or facts.

The strong CPU dictionary receives the same raw bit stream and constant training
schedule. It also allocates all16 histories and recalls every held-out window.
There is no execution-speed, energy, CPU-contention or unique computability claim.

## What runs on the chip

Three sequential FNP3s: fixed conjunction encoder[1,3,0], context bank[2,1,0],
fixed output relay[2,2,0]. The context bank has32 initially random plastic records
and8 fixed raw-history helper records. Each has32 binary connections. The fixed
helpers use a learning threshold32, so the strict greater-than eligibility test
cannot select them; their complete captured RAM stays unchanged throughout.
They decode raw history from the engineered conjunction basis. No arbitrary
context-to-record mapping is planted in the32 plastic records.

The host initially seeds three zero-valued past symbols. Afterwards it writes
only the16 current-symbol channels into the preceding native output payload.
The next descriptor reads that output buffer directly. Native helpers supply the
next retained history. The compiler, returned packet, causal pointers, all input
bytes, fixed encoder/relay RAM and every learning-bank boundary are checked.
The current input contains only a raw binary symbol; all training labels are the
same0 and leave every plastic record in the same competing class. During learning,
the native eligible-record winner chooses which connections to update. During
recall, the native activation mask has exactly one responding learned record.

The host still schedules every forward and explicitly switches learning off for
queries. It computes expected traces for verification but never repairs a weight,
selects a context record or reconstructs the acting history. This is a bounded,
host-paced feedback circuit. It does not keep running indefinitely with Nano quiet.

## Stages and controls

Seed113101 native1,2,4,12-cycle stages passed full output/RAM replay. They recalled
4,12,16,16 distinct histories respectively; only the final stage required all16.
After first-seed frozen, untrained and history-severed controls passed, four more
initializations ran native/frozen/untrained12-cycle arms. Original four-history
training qualified before the new-history expansion in each trained arm.

The first-seed history ablation clears only carried history during probes. It
returns no learned-context responses with identical captured learning RAM to the
intact arm. This isolates required carried input from changed stored knowledge.

Only after all main seeds passed, six additional resident sessions appended a
prospectively sealed128-bit held-out stream: five native, one trained-frozen. Each
new run's entire323-event prefix matches its original run before the new suffix.
All seven rehashed semantic corruptions (labels, history, RAM, mode, response,
fixed frontend and source pointer) are rejected by the portable verifier.

**25 qualified physical sessions /8,379 verified forwards.** Two earlier physical
prefixes failed after1 and20 captured forwards and remain archived: total physical
capture8,400 forwards across27 admissions. Two CPU audit failures, two superseded
CPU-reference traces and two compiler failures remain separate from qualified data.
No failed attempt was overwritten or silently added to scientific successes.

## Failures that changed the implementation

- Direct two-input FullyConnected merge rejected by the SDK; Concatenate reported
  incompatible IP version. Neither opened hardware. The admitted sequential fixed
  helpers replaced that proposed branch, with a prospective amendment.
- The unprojected dual-rail CPU screen produced only8–11 distinct response masks
  for16 presented histories across five seeds. The chosen fixed combination basis
  passes. Input density also differs, so this is an exploratory rejected design,
  not a controlled causal proof of one encoding's superiority.
- The first one-class hardware query changed connections when global learning was
  enabled despite omitting a label. Query-mode gating was corrected explicitly.
  This is consistent with single-class unsupervised learning; it is not a newly
  discovered silicon instruction or undocumented capability claim.
- The next first-teaching edit disagreed with a CPU model that advanced its learning
  selector on disabled-mode queries. A mode-specific reference leaves that state
  unchanged. All subsequent complete traces, including mode transitions, match.
  The hidden register itself was not read; complete learning-state restoration
  and electrical spike traces remain unqualified.
- One CPU prefix audit applied the final endpoint too early; one held-out audit
  failed JSON serialization after its numeric assertions. Both receipts persist.
  Amendments01–07 describe the corrections before dependent physical admissions.

## Limits and next gate

This qualifies native **allocation and selection over fixed features**. It does
not learn the feature basis, invent semantic task contexts, predict consequences,
choose actions, repair physical damage, establish power-loss persistence, or prove
an efficiency advantage. Three NPUs and32 plastic records do not exhaust the chip.
Single-class unsupervised learning is already part of the public SDK; this is new
integration and verification in this project, not a claim to invent that primitive.

The next admitted capability should be a downstream native consequence learner
consuming these learned activations directly, with no host context-ID lookup. It
must predict before feedback, adapt to changed outcomes, retain unchanged contexts,
and pass a fresh sealed ordering against trained-frozen and strong CPU controls.
A full useful loop and a measured substrate benefit remain separate future gates.
The incremental films remain research evidence; this component alone is not the
promised next major standalone X milestone. Nothing was posted to X.

## Reproduce the public audit

```sh
python -m discovery.archive audit
python -m discovery.audit_mutations
```

[Numeric audit](../../results/20261024-context-discovery.audit.json) ·
[public tooling](../../discovery/README.md) ·
[official learning semantics](https://doc.brainchipinc.com/user_guide/akida.html) ·
[official unsupervised example](https://doc.brainchipinc.com/examples/edge/plot_2_edge_learning_parameters.html)
