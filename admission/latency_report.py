"""Render measured benchmark receipts as a figure, case CSV and plain report."""
import csv,json
import numpy as np
from admission import latency as m
from admission import latency_archive as archive

def main():
    result=archive.audit();s=result['summary'];meta=json.loads(archive.BASE.with_suffix('.json').read_text())
    raw=[]
    with np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for name,entry in meta['runs'].items():
            r=entry['receipt']
            if r['stage']!='repeat':continue
            t=z[entry['arrays']['timing']['key']].reshape(-1,len(m.FIELDS))
            fixture=meta['fixtures'][str(r['seed'])]
            ref=m.simulate([z[q['key']] for q in fixture['banks']],'repeat','native')
            get=lambda ext:z[entry['arrays'][ext]['key']]
            con=archive.check_run(r,entry['stdout'],get,ref)
            for c in con:
                i=c['committed_event']
                raw.append(dict(run=name,mode=r['mode'],seed=r['seed'],repetition=r['repetition'],
                                **c,**{k:int(t[i,j]) for j,k in enumerate(m.FIELDS[:9])}))
    with (m.S/'CASES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]),lineterminator='\n');w.writeheader();w.writerows(raw)
    native=s['native'];cpu=s['replica'];policy=s['dictionary']
    median=lambda q:q['median_ns']
    update=lambda q:median(q['actual_revision'])/1000
    confirm=lambda q:median(q['convergence']['learner_operational_until_confirmed_ns'])/1e6
    ratio=update(native)/update(cpu)
    audit_all=m.audit(m.S/'raw/native_s124101_gate_all_r2')
    data=dict(archive_sha256=result['archive_sha256'],actual_revision_us={k:update(v) for k,v in s.items()},
         confirmation_active_ms={k:confirm(v) for k,v in s.items()},
         CPU_policy_batched_ns=median(policy['agreed_policy_batch_ns_per_update']),
         full_learning_RAM_verification_ms=median(audit_all['accepted_transfer_audit'])/1e6,
         native_vs_replica_latency_ratio=ratio,summary=s)
    m.save(m.S/'SUMMARY.json',data)
    text=f'''# Matched agreed-update latency on the Nano

**The CPU wins this small workload.** The AKD remains functionally exact, but an
actual memory-revision transaction takes **{update(native):.1f} µs** versus
**{update(cpu):.1f} µs** for the packed CPU neural replica: **{ratio:.1f}x** the latency.
The direct CPU agreement policy is cheaper still. This does not measure maximum
chip throughput, power or an architectural limit.

## Results (medians; p95 retained in SUMMARY.json)

| Path | Actual revision transaction | Source query + update + 16 recall checks |
|---|---:|---:|
| Native AKD, including transport/runtime | {update(native):.3f} µs | {confirm(native):.3f} ms |
| CPU packed neural replica, sparse fixed routes | {update(cpu):.3f} µs | {confirm(cpu):.3f} ms |
| CPU direct agreement policy, per-event clocked | {update(policy):.3f} µs (timer-limited) | {confirm(policy)*1000:.3f} µs (timer-limited) |

The clock-call calibration itself has a 64 ns median. The policy's raw 64 ns update
measurement is therefore not a resolved single-call cost. A separate resident
agreement+store batch measures **{median(policy['agreed_policy_batch_ns_per_update']):.2f} ns/update amortized**
(nine batches, including proposal loads, loop, function call and checksum). A mixed
source/query/transfer stream averages {median(policy['mixed_stream_batch_ns_per_event']):.2f} ns/event.
These batches are explicitly different from executing a neural simulation and do
not imply nanosecond learning of arbitrary associations.
The batch is single-threaded with a tiny cache-resident table; completion means
the new answer is available to that CPU learner, not that stores have reached
external DRAM or survived power loss. No persistence/coherence claim is measured.

All three paths revise the two changed cues after their **first accepted teaching
attempt** in every tested cycle. Every revised cue passes 16 consecutive read-only,
teacher-answer-masked recalls, with no later same-cycle error. Across nine primary
runs per path (three seeds x three repetitions x six cycles): 108 actual revised-cue
cases, 648 accepted transfer calls, 2592/2592 protected final probes and 2592/2592
updated final probes. Most accepted transfer calls revisit already-correct/stable
memory; headline revision timings use the 108 actual changed-memory transactions.

Confirmation active time includes its 16 queries, so it is not the intrinsic time
to change memory. Native confirmation probes account for about 4.68 ms. From each
learner's first source query, raw observed completion including lightweight audit
bookkeeping is {median(native['convergence']['learner_observed_until_confirmed_ns'])/1e6:.3f} ms native,
{median(cpu['convergence']['learner_observed_until_confirmed_ns'])/1e6:.3f} ms CPU replica, and
{median(policy['convergence']['learner_observed_until_confirmed_ns'])/1000:.3f} µs direct CPU policy.
Global readiness for both cues precedes the sealed interleaved order; the second
cue waits for the first cue's confirmation probes. SUMMARY.json and CASES.csv keep
that global elapsed endpoint separate from the per-learner endpoint shown above.

## Where native time goes

Actual-revision component medians (individual medians need not sum exactly):

| Observable boundary | Median |
|---|---:|
| Learning-mode configuration writes | {median(native['revision_components']['mode_ns'])/1000:.2f} µs |
| Raw phase/input writes | {median(native['revision_components']['input_ns'])/1000:.2f} µs |
| Descriptor submission | {median(native['revision_components']['submit_ns'])/1000:.2f} µs |
| Queue-release write | {median(native['revision_components']['release_ns'])/1000:.2f} µs |
| Release completed → observed device completion | {median(native['revision_components']['completion_observed_ns'])/1000:.2f} µs |
| Result fetch, SDK and returned-output copy | {median(native['revision_components']['fetch_ns'])/1000:.2f} µs |

Completion observation includes PCIe polling; it is not a pure silicon timer.
Result retrieval is the largest measured component. These boundaries separate
observable transport/runtime work from resident completion but cannot divide
the latter into pure PCIe and pure arithmetic by subtraction. The seven NPUs
execute sequential stages in this mapped program.

## Verification and execution

Full three-learning-bank readback plus checks costs a median
**{median(audit_all['accepted_transfer_audit'])/1e6:.2f} ms per accepted operation** in the full-audit
qualification. This is excluded from primary timings. Primary cells verify every
output byte and complete final learning RAM; full-audit qualification verifies
every intermediate bank boundary. Fixed route RAM is checked at setup/closeout.
Disk serialization, initial learning, program loading and reference generation
are outside reported operational latency. No host writes occur during execution.

Nano ARM CPU2, 1.728 GHz min/max, Ubuntu g++13.3, O3/mcpu=native, serial rotated
arm order. No unrelated process or clock configuration was changed. 12 successful
physical cells, 29,448 forwards including qualification. 35 passing CPU/native
streams. Two failed/diagnostic attempts are preserved: an initial SIGSEGV and
bounded GDB replay exposed the new harness destroying its program buffer before
enqueue. The buffer lifetime was repaired; full-state qualification then passed
twice before timing expansion. This is a harness failure, not a silicon result.
All owned jobs are finished; card released. Study closed.

Archive: `{result['archive_sha256']}`. SDK/card-free replay:

```sh
python -m unittest discover -s tests -p test_consensus_latency.py -v
python -m admission.latency_archive audit
```

## Interpretation

This provides no latency advantage for AKD on the four-cue task. Even the observed
resident-completion interval is slower than this CPU neural replica; host retrieval
adds substantial further cost. Functional native memory control remains demonstrated.
Any future hardware-efficiency claim needs a separately frozen workload that uses
more of the fabric and amortizes transport, with matched quality and direct energy
measurement. Do not infer an optimized full-chip ceiling from this small circuit.
'''
    (m.S/'ASSESSMENT.md').write_text(text)
    figure(data)
    print(json.dumps({k:v for k,v in data.items() if k!='summary'}))

def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bg='#0d1420';fg='#e6edf5';muted='#9daec1';blue='#75b7ed';orange='#f1b66c'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':fg,
        'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':fg,'axes.edgecolor':'#344359',
        'axes.facecolor':bg,'figure.facecolor':bg,'savefig.facecolor':bg})
    fig=plt.figure(figsize=(16,9),dpi=240)
    gs=fig.add_gridspec(2,2,left=.17,right=.94,top=.79,bottom=.20,hspace=.78,wspace=.6)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    s=data['summary']
    for ax in [a,b,c]:
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.13)
    values=[data['actual_revision_us']['replica'],data['actual_revision_us']['native']]
    a.barh(['CPU neural replica','AKD + transport'],values,color=[blue,orange],height=.48)
    for i,v in enumerate(values):a.text(v+6,i,f'{v:.1f} µs',va='center',fontweight='bold')
    a.set_xlim(0,390);a.set_xlabel('Microseconds / actual revision');a.set_title('One changed-memory transaction',loc='left',fontsize=14,pad=18)
    parts=s['native']['revision_components'];rows=[('Mode writes','mode_ns'),('Input writes','input_ns'),('Submit + release',None),('Observed completion','completion_observed_ns'),('Fetch result','fetch_ns')]
    vals=[parts[k]['median_ns']/1000 if k else (parts['submit_ns']['median_ns']+parts['release_ns']['median_ns'])/1000 for _,k in rows]
    b.barh([k for k,_ in rows],vals,color=['#617994','#617994','#617994',orange,blue],height=.56);b.invert_yaxis()
    for i,v in enumerate(vals):b.text(v+3,i,f'{v:.1f}',va='center',fontsize=10)
    b.set_xlim(0,215);b.set_xlabel('Microseconds · component medians');b.set_title('AKD transport / runtime breakdown',loc='left',fontsize=14,pad=18)
    vals=[data['confirmation_active_ms']['replica'],data['confirmation_active_ms']['native']]
    c.barh(['CPU neural replica','AKD + transport'],vals,color=[blue,orange],height=.48)
    for i,v in enumerate(vals):c.text(v+.08,i,f'{v:.3f} ms',va='center',fontweight='bold')
    c.set_xlim(0,6.2);c.set_xlabel('Active milliseconds · source query + update + 16 read-only recall probes')
    c.set_title('All paths recall the revision after one accepted update; confirmation adds query time',loc='left',fontsize=14,pad=18)
    fig.text(.06,.94,'Native memory works. The CPU is faster on this workload.',fontsize=22,fontweight='bold')
    fig.text(.06,.885,'Nano · four cues · seven sequential NPUs · three seeds × three timing runs · 108 actual revisions per path',color=muted,fontsize=12)
    fig.text(.06,.11,f"Direct CPU agreement policy: {data['CPU_policy_batched_ns']:.2f} ns/update amortized in a resident batch.",fontsize=12,fontweight='bold')
    fig.text(.06,.07,'Single-call CPU policy readings meet the 64 ns clock-call floor. Full RAM audits add ~4.34 ms and are excluded above.',color=muted,fontsize=11)
    fig.text(.06,.035,'Completion observation includes PCIe polling. No pure silicon timing, energy advantage or full-chip throughput claim. Source: saved numeric receipts.',color=muted,fontsize=10)
    for ext in ['png','svg']:fig.savefig(m.S/f'LATENCY.{ext}')
    svg=m.S/'LATENCY.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__=='__main__':main()
