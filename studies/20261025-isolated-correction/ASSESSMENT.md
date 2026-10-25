# Two resident memories correct themselves without cross-context RAM changes

Completed31 physical cells/31,500 native forwards. All returned bits, full RAM
boundaries, fixed route readbacks, descriptor labels and source pointers replay
against the independent Python and packed C++ references. No execution failures
occurred. Eight deliberate semantic corruptions are rejected.

## Measured qualification

Both banks are learning-enabled on the same teacher clocks. Native feature routing
selects the receiving context; the host never switches banks according to context.
The same four cue IDs initially have opposite answers in the two contexts.

| Arm | Read-only reversal probes | All online queries | Connection replacements including bootstrap |
|---|---:|---:|---:|
| Native error + retirement | 6912/6912 | 7327/7536 | 1316 |
| No retirement path | 4368/6912 | 4694/7536 | 613 |
| Trained then frozen | 5184/6912 | 5609/7536 | 653 |
| CPU latest-fact dictionary | 6912/6912 | 7344/7536 | not comparable |
| CPU eight-entry LRU per context | 6912/6912 | 7344/7536 | not comparable |

Six native replication cells: three initial connection seeds on the base stream;
the first initialization also receives three shuffled-order replications. This is
not a full three-by-four factorial. Twelve alternating context reversals per cell;
96 probes/reversal. Source seeds and stream IDs are in the protocol and numeric audit.

Every foreign-context bank boundary stays byte-identical:8400 boundaries across
the six native runs. Twenty-four complete original records (36 words each across
both contexts/runs) retain their original payloads. These counts concern captured
state, not unobserved internal registers. First-seed solo A and solo B each match
the corresponding filtered mixed-run RAM trajectory at all700 own-context boundaries,
with identical final RAM. Freezing A reduces its probes to432/576 while B remains
576/576; freezing B produces the symmetric result.

## Capacity gate failed

First-seed pressure loads eight new cue IDs into A while B remains untouched.
A's final recall is48/96 under native retirement,80/96 without retirement and32/96
when trained-frozen. B remains96/96 in all three. The native useful-capacity gate
required A>=87/96 and B=96/96: it failed. No further pressure seeds were admitted.
The CPU dictionary scores192/192 combined; eight-entry LRU160/192. Context isolation
therefore does not solve allocation/eviction or overloaded-bank usefulness. The
retirement mechanism that helps contradictory facts can harm this capacity regime.

## Claim boundary and next direction

Seven sequential FNP3s contain16 plastic records and80 fixed helper records. Native
comparison/routing and weight updates are qualified under engineered shared/key
coding and phase rules. Host supplies raw contexts, cues, external actual/complement
labels and fixed query/positive/negative clocks. Mode histories and host counters
are instrument journals, not electrical CSR traces. Final response bits and RAM are
read back; intermediate potentials/features are derived.

The result is functional resident isolation and selective correction, not electrical
concurrency, general memory management, an autonomous scheduler, persistence through
power loss, or a speed/energy advantage. Strong CPU controls also pass and acquire
slightly better. The study is closed, owned jobs are empty and the device was free.

The next fresh study tests learned-source-to-student transfer with constant host
transfer labels and teacher-silent exams. That hypothesis is separate from these
completed results and needs its own staged physical gates.

```sh
python -m admission.isolation_archive audit
python -m admission.isolation_mutations
```
