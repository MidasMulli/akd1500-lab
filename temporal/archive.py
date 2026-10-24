"""Portable audit of recorded temporal inputs, native outputs and learning RAM."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from temporal.model import design,simulate,SEEDS,ARMS
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-temporal-context';BASE=P/'results/20261024-temporal-context'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def masks(v):return np.unpackbits(np.asarray(v,dtype='<u4')[...,2:].reshape(*v.shape[:-1],8,12)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def build():
    arrays={};fixtures={};runs={}
    for d in sorted((S/'fixtures01').glob('s*')):
        fs={}
        for name in ['codes','cues','weights','old','new','changed','initial.ram','fixed.ram']:
            t='u1' if name in ['codes','cues'] else 'i1' if name=='weights' else '<i4' if name in ['old','new','changed'] else '<u4';key=d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(d/name,t);fs[name]=dict(key=key,sha256=sha(d/name))
        fixtures[d.name]=dict(arrays=fs,metadata=json.loads((d/'BUILD.json').read_text()))
    for d in sorted((S/'raw').iterdir()):
        fs={}
        for p in sorted(d.glob('result.*')):
            name=p.name[7:];t='u1' if name in ['inputs','scores','task_scores','stateless_scores'] else '<i8' if name=='delays_ns' else '<u4';key=d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
        raw=(d/'stdout').read_text();runs[d.name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),arrays=fs,stdout=json.loads(raw) if raw.startswith('{') else raw,stderr=(d/'stderr').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,source_freezes={p.name:json.loads(p.read_text()) for p in sorted((S/'evidence').glob('FROZEN*.json'))},scope='Native output carries context; host supplies only current symbol except explicit history ablation. Seeded synthetic task; externally labelled, serial host-paced events; CPU state machine also succeeds.'))
    return audit()
def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];failures=[];main={};cpu={}
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
      for b in [*m['fixtures'].values(),*m['runs'].values()]:
        for q in b['arrays'].values():assert hashlib.sha256(z[q['key']].tobytes()).hexdigest()==q['sha256']
      for name,run in m['runs'].items():
        r=run['receipt'];get=lambda k:z[run['arrays'][k]['key']]
        if r['exit_code']:
            failures.append(dict(name=name,error=run['stderr'],completed_forwards=len(get('scores'))//256 if 'scores' in run['arrays'] else 0));continue
        seed,arm,epochs=r['seed'],r['arm'],r['epochs'];f=m['fixtures'][f's{seed}'];fg=lambda k:z[f['arrays'][k]['key']];initial=fg('initial.ram');codes,cues,w,old,new,changed=design(seed)
        for k,v in [('codes',codes),('cues',cues),('weights',w),('old',old),('new',new),('changed',changed)]:assert np.array_equal(fg(k),v.ravel())
        a=simulate(initial,seed,arm,epochs);events=a['events'];count=len(events);x=get('inputs').reshape(count,256);y=get('scores').reshape(count,256);states=get('states').reshape(count,98)
        assert np.array_equal(x,a['inputs']) and np.array_equal(y,a['outputs']) and np.array_equal(states,a['states']) and np.array_equal(get('ram'),states[-1])
        assert np.all(masks(states).sum(-1)==32)
        for i,e in enumerate(events):
            assert np.array_equal(x[i,128:],cues[e['cue'],128:])
            if i and not(arm=='severed' and e['phase']=='probe' and e['cue']>=4):assert np.array_equal(x[i,:128],y[i-1,:128])
            if arm not in ['native','severed'] or e['label']<0:assert np.array_equal(states[i],states[i-1] if i else initial)
        probe=[e for e in events if e['phase']=='probe'];pred=[e for e in probe if e['cue']>=4];ctx=[e for e in probe if e['cue']<4];changed_es=[e for e in pred if e['context'] in changed];stable=[e for e in pred if e['context'] not in changed]
        correct=lambda es,yy:sum(np.array_equal(yy[e['index']],codes[e['target']]) for e in es)
        assert correct(ctx,y)==len(ctx)
        if arm=='native':assert correct(pred,y)==len(pred)
        if arm in ['frozen','unlabelled']:assert correct(stable,y)==len(stable)
        if arm=='severed':assert not y[[e['index'] for e in pred]].any()
        before=masks(initial);replacements=0
        for state in states:
            after=masks(state);replacements+=int(((after==1)&(before==0)).sum());before=after
        row=dict(name=name,mode=r['mode'],seed=seed,arm=arm,epochs=epochs,events=count,changed_correct=correct(changed_es,y),changed_total=len(changed_es),stable_correct=correct(stable,y),stable_total=len(stable),context_correct=correct(ctx,y),context_total=len(ctx),cumulative_replacements=replacements)
        if r['mode']=='native':
            o=run['stdout'];assert r['device_free'] and o['program_calls']==1 and o['weight_restore_calls']==o['threshold_resets']==0 and o['cleanup']=='unprogrammed' and o['current_half_writes']==count
            assert o['history_ablation_writes']==sum(arm=='severed' and e['phase']=='probe' and e['cue']>=4 for e in events)
            assert np.array_equal(get('initial.ram'),initial)
            for tag in ['initial','final']:assert np.array_equal(get('fixed.'+tag),fg('fixed.ram'))
            desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');delay=get('delays_ns');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
            for i,e in enumerate(events):
                label=e['label'] if arm in ['native','severed'] else -1;assert ((int(desc[i,8])>>16)&1023)==(label+1 if label>=0 else 0);assert delay[i]>=e['delay_ms']*1_000_000
                assert 0x20000100<=inp[i] and int(inp[i])+256<=0x20100000
                if i:assert inp[i]==addr[i-1]+32 and addr[i]!=addr[i-1]
                else:assert not x[i,:128].any()
                if arm=='severed' and e['phase']=='probe' and e['cue']>=4:assert not x[i,:128].any()
                assert np.array_equal(get('packet'+str(i)).view('u1')[32:],y[i])
            if epochs==8:main[(seed,arm)]=(row,states.copy(),y.copy(),x.copy(),events)
        else:
            task=get('task_scores').reshape(count,256);stateless=get('stateless_scores').reshape(count,256);assert np.array_equal(task,a['task_outputs']) and np.array_equal(stateless,a['stateless_outputs'])
            row.update(task_correct=correct(pred,task),stateless_correct=correct(pred,stateless),prediction_total=len(pred))
            if arm=='native':assert row['task_correct']==len(pred)
            if epochs==8 and arm=='native':cpu[seed]=row
        rows.append(row)
      assert set(main)=={(s,a) for s in SEEDS for a in ARMS} and set(cpu)==set(SEEDS)
      for seed in SEEDS:
        native=main[(seed,'native')];severed=main[(seed,'severed')];events=native[4];ids=[e['index'] for e in events if e['phase']=='probe'];assert np.array_equal(native[1][ids],severed[1][ids]),'identical learning RAM in history ablation'
        query_ids=[e['index'] for e in events if e['phase']=='probe' and e['cue']>=4]
        for control in ['frozen','unlabelled']:assert np.array_equal(native[3][query_ids],main[(seed,control)][3][query_ids]),'matched complete probe inputs across learning controls'
        for epoch in range(8):
          for variant in ['full','short','long']:
            ii=[e['index'] for e in events if e['phase']=='probe' and e['epoch']==epoch and e['variant']==variant and e['cue']>=4]
            assert len(ii)==4 and len({native[2][i].tobytes() for i in ii})==4
            assert all(np.array_equal(native[3][i,128:],native[3][ii[0],128:]) for i in ii)
    physical=[r for r in rows if r['mode']=='native'];assert len(physical)==23 and sum(r['events'] for r in physical)==5400
    result=dict(status='INDEPENDENT_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=len(physical),verified_forwards=sum(r['events'] for r in physical),failed_attempts=failures,seeds=SEEDS,changes_per_run=8,primary={a:{k:sum(main[(s,a)][0][k] for s in SEEDS) for k in ['changed_correct','changed_total','stable_correct','stable_total','context_correct','context_total','cumulative_replacements']} for a in ARMS},cpu_state_machine=dict(correct=sum(cpu[s]['task_correct'] for s in SEEDS),total=sum(cpu[s]['prediction_total'] for s in SEEDS)),symbol_only_ablation=dict(correct=sum(cpu[s]['stateless_correct'] for s in SEEDS),total=sum(cpu[s]['prediction_total'] for s in SEEDS)),probe_learning_RAM_identical_with_history_ablation=True,rows=rows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
