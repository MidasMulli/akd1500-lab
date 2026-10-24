"""Scientific evidence chart; compiler plans never become measured circuitry."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from resident.archive import P, BASE, audit_public
from resident.design import Network, schedule


def main():
    audit_public();m=json.loads(BASE.with_suffix('.json').read_text())
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        r=next(r for r in m['runs'] if r['name']=='smoke_b2_s99101_repair');prefix=r['name']+'__'
        net=Network([z[prefix+f'initial{k}'].tolist() for k in range(2)])
        x,rows=schedule(2,'repair',True)
        for i in range(129):
            before=[b.ram[:] for b in net.banks];net.forward(x[i],rows[i]['label'],rows[i]['mask'])
        pair=r['snapshot_coordinates'].index([128,1]);observed=z[prefix+'states'][pair]
        previous=np.asarray(before[1][2:],np.uint32).reshape(64,20)
        expected=np.asarray(net.banks[1].ram[2:],np.uint32).reshape(64,20)
        actual=observed[2:].reshape(64,20)
        expected_ids=np.flatnonzero((previous!=expected).any(1)).tolist()
        actual_ids=np.flatnonzero((previous!=actual).any(1)).tolist()
    evidence=dict(physical_sessions=6,observed_forwards=663,full_bank_captures=796,
        one_bank_successes=[dict(attempt=i,events=176) for i in [1,2,3]],
        mixed_attempts=[dict(attempt=i,first_divergence_event_1_based=e,completed_exact_prefix=e-1) for i,e in [(1,129),(2,1),(3,5)]],
        attempted_mapping=dict(plastic_banks=6,plastic_records=384,resident_npus=17,status='COMPILER_ONLY_NOT_RUN'),
        first_downstream_mismatch=dict(attempt=1,event_1_based=129,expected_changed_neurons=expected_ids,observed_changed_neurons=actual_ids),
        boundary='Mixed-resource adapter not validated. Existing two-bank FNP-only result remains separate. No CPU or electrical advantage.')
    v=P/'visual';(v/'resident-boundary-data.json').write_text(json.dumps(evidence,indent=2)+'\n')
    bg='#0a111b';panel='#111e2c';ink='#edf2f8';muted='#99aec3';blue='#78c9ed';orange='#efb76c'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':ink,'axes.labelcolor':ink,'xtick.color':muted,'ytick.color':muted,'font.size':16})
    fig=plt.figure(figsize=(16,9),dpi=240,facecolor=bg)
    fig.text(.06,.932,'AKD1500 / RESIDENT LEARNING BOUNDARY',size=15,color=muted)
    fig.text(.06,.864,'The compiler maps it. The state checks stop it.',size=28,weight='bold')
    fig.text(.06,.814,'6 owned sessions  ·  663 actual forwards  ·  796 full bank captures  ·  larger stages not admitted',size=15,color=muted)
    a=fig.add_axes([.06,.43,.4,.32],facecolor=panel);a.set_axis_off()
    a.text(.05,.85,'ONE BANK / EXACT OUTPUT + MEMORY',size=14,color=blue,weight='bold')
    for j in range(3):
        a.text(.06,.65-j*.16,f'Implementation attempt {j+1}',size=14)
        a.text(.94,.65-j*.16,'176 / 176',ha='right',size=16,color=blue,weight='bold')
    a.text(.05,.08,'Same seed + task; repeats are mechanics checks.',size=12,color=muted)
    a=fig.add_axes([.5,.43,.44,.32],facecolor=panel);a.set_axis_off()
    a.text(.05,.85,'TWO BANKS / FIRST RAM DIVERGENCE',size=14,color=orange,weight='bold')
    for j,event in enumerate([129,1,5]):
        a.text(.06,.65-j*.16,f'Adapter attempt {j+1}',size=14)
        a.text(.94,.65-j*.16,f'input {event}',ha='right',size=16,color=orange,weight='bold')
    a.text(.05,.08,'Failures retained. No four- or six-bank hardware run.',size=12,color=muted)
    a=fig.add_axes([.06,.17,.4,.21],facecolor=panel);a.set_axis_off()
    a.text(.05,.8,'FIRST DOWNSTREAM UPDATE / ATTEMPT 1',size=13,color=muted,weight='bold')
    for row,(ids,label,color) in enumerate([(expected_ids,'Reference',blue),(actual_ids,'Measured RAM',orange)]):
        yy=.49-row*.24;a.text(.05,yy,label,size=13,va='center')
        for j in range(64):
            a.plot(.36+(j%16)*.036,yy+.032-(j//16)*.022,'o',ms=3.2,color=color if j in ids else '#34485a')
        a.text(.94,yy,'N'+','.join(map(str,ids)),ha='right',size=12,color=color,va='center')
    a.set_xlim(0,1);a.set_ylim(0,1)
    a.text(.05,.045,'Each dot = one captured learning-neuron record.',size=11,color=muted)
    a=fig.add_axes([.5,.17,.44,.21],facecolor=panel);a.set_axis_off()
    a.text(.05,.8,'PLANNED SCALE / COMPILER EVIDENCE ONLY',size=13,color=muted,weight='bold')
    a.text(.05,.47,'6 plastic banks · 384 records · 17 NPUs',size=19,weight='bold')
    a.text(.05,.18,'Not a demonstrated six-bank learner. Not a chip limit.',size=12,color=orange)
    fig.text(.06,.105,'The new FNP3 → fixed CNP communication path requires a validated learning-state adapter.',size=15)
    fig.text(.06,.065,'Earlier two-bank FNP-only learning remains valid. No new CPU-efficiency, energy or autonomous-learning claim.',size=12,color=muted)
    fig.savefig(v/'resident-boundary-4k.png',facecolor=bg);fig.savefig(v/'resident-boundary.svg',facecolor=bg);plt.close(fig)
    files=[BASE.with_suffix('.json'),BASE.with_suffix('.npz'),v/'resident-boundary-data.json',v/'resident-boundary-4k.png',Path(__file__)]
    (v/'RESIDENT_BOUNDARY_SOURCE_BINDING.json').write_text(json.dumps(dict(status='NATIVE_FAILURE_AND_COMPILER_ONLY_BOUND',
        hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')


if __name__=='__main__':
    from pathlib import Path
    main()
