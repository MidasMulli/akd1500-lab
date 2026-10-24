# Preserved near-budget off-target update

Status: **FAILED_PRESERVED_EXPANSION_HALTED**. All owned jobs are stopped; the device is released.

Frozen inference passes at 11,264 records. Learning fails at its fourth forward: potentials still match, but intended record 8849 remains inconsistent with the expected update and actual changes affect records 2295/2296. The first three full states are exact. The failed prefix, raw state and expected state are retained; no completed timing vector was saved. Expansion stops. Candidate full alias explanations fail and remain rejected hypotheses.

Observed totals: 2 physical sessions, 12 forwards and 12 complete record snapshots; 80 CPU sample executions. Failed counts are saved prefix lengths, never intended run lengths.

See [the combined assessment](../../coprocessor/FNP2.md), [independent audit](../../results/2026-10-24-fnp2-audit.json) and [numeric evidence](../../results/2026-10-24-fnp2-contract.json). Licensed runtime sources and SDK-generated programs remain private.
