"""Portable raw audit, including superseded CPU forecasts and failed prefix."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from plasticity.model import simulate,design,SEEDS,ARMS,H
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-resident-adaptation';BASE=P/'results/20261024-resident-adaptation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):Path(p).write_text(json.dumps(a,indent=2)+'\n')
def masks(a):return np.unpackbits(np.asarray(a,dtype='<u4')[...,2:].reshape(*a.shape[:-1],8,12)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def build():
    arrays={};fixtures={};runs={};fr1=sha(S/'evidence/FROZEN01.json')
    for d in sorted((S/'fixtures02').glob('s*')):
        fs={}
        for p in sorted(d.iterdir()):
            if p.name not in ['codes','partial','old','new','changed'] and p.suffix not in ['.ram','.fixed']:continue
            key=d.name+'__'+p.name.replace('.','_');t='u1' if p.name in ['codes','partial'] else '<i4' if p.name in ['old','new','changed'] else '<u4';arrays[key]=np.fromfile(p,t);fs[p.name]=dict(key=key,sha256=sha(p))
        fixtures[d.name]=dict(metadata=json.loads((d/'BUILD.json').read_text()),arrays=fs)
    for d in sorted((S/'raw').iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text());fs={}
        for p in sorted(d.glob('result.*')):
            k=p.name[7:];t='u1' if k in ['inputs','scores','table_inputs','table_scores'] else '<i8' if k in ['quiet','latency_ns'] else '<i4' if k=='state_ids' else '<u4';key=d.name+'__'+k.replace('.','_');arrays[key]=np.fromfile(p,t);fs[k]=dict(key=key,sha256=sha(p))
        stdout=(d/'stdout').read_text();runs[d.name]=dict(receipt=r,arrays=fs,stdout=json.loads(stdout) if stdout.strip().startswith('{') else stdout,stderr=(d/'stderr').read_text(),superseded_cpu_model=r['mode']=='cpu' and r['freeze_sha256']==fr1)
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    metadata=dict(program_identity=json.loads((S/'evidence/PROGRAM_IDENTITY01.json').read_text()),superseded_reference_source=(S/'failed_attempts/source01/model.py').read_text(),superseded_own_source_hashes={q.name:sha(q) for q in (S/'failed_attempts/source01').iterdir() if q.is_file()},archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,source_freezes={p.name:json.loads(p.read_text()) for p in sorted((S/'evidence').glob('FROZEN*.json'))},repairs=json.loads((S/'evidence/REPAIRS.json').read_text()),selector_diagnostic=json.loads((S/'evidence/SELECTOR_DIAGNOSTIC01.json').read_text()),selector_causal=json.loads((S/'evidence/SELECTOR_CAUSAL01.json').read_text()),scope='Repeated external observation, native update and SRAM feedback in a host-paced resident model; two alternating synthetic permutations, not indefinite self-scheduling or electrical efficiency.')
    save(BASE.with_suffix('.json'),metadata);return audit()
def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];superseded=[];failures=[];selector={};timings=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
      for b in [*m['fixtures'].values(),*m['runs'].values()]:
        for q in b['arrays'].values():assert hashlib.sha256(z[q['key']].tobytes()).hexdigest()==q['sha256']
      for name,run in m['runs'].items():
        r=run['receipt'];get=lambda k:z[run['arrays'][k]['key']]
        if run['superseded_cpu_model']:superseded.append(name);continue
        if r['exit_code']:
            failures.append(dict(name=name,error=run['stderr'].strip(),observed_completed_forwards=len(get('scores'))//256,source_frozen=True));continue
        seed,arm,epochs,q=r['seed'],r['arm'],r['epochs'],r.get('extra_queries',0);full=r.get('full_queries',False);f=m['fixtures'][f's{seed}'+(f'_q{q}'+('full' if full else '') if q else '')];fg=lambda k:z[f['arrays'][k]['key']];initial=fg(arm+'.ram');a=simulate(initial,seed,arm,epochs,q,full);events=a['events'];count=len(events);codes,partial,old,new,changed=design(seed)
        assert np.array_equal(codes,fg('codes').reshape(8,256)) and np.array_equal(partial,fg('partial').reshape(8,256)) and np.array_equal(old,fg('old')) and np.array_equal(new,fg('new')) and np.array_equal(changed,fg('changed'))
        y=get('scores').reshape(count,256);x=get('inputs').reshape(count,256);ids=get('state_ids');states=get('states').reshape(-1,98);assert len(ids)==len(states) and np.all(np.diff(ids)>0);assert np.array_equal(y,a['outputs']) and np.array_equal(x,a['inputs']);assert np.array_equal(states,a['states'][ids]) and np.array_equal(get('ram'),a['states'][-1]);assert np.all(masks(states).sum(-1)==32)
        before=masks(initial);replacements=0;update_events=0
        for state in states:
            after=masks(state);n=int(((after==1)&(before==0)).sum());replacements+=n;update_events+=n>0;assert n==int(((after==0)&(before==1)).sum());before=after
        epoch_rows=[]
        for e in range(epochs):
            truth=new if e%2==0 else old;qr=[v['index'] for v in events if v['kind']=='query' and v['epoch']==e];fr=[v['index'] for v in events if v['kind']=='feedback' and v['epoch']==e];correct=np.array([np.array_equal(y[i],codes[truth[events[i]['context']]]) for i in qr]);consistent=0
            for i in fr:
                s=next((j for j in range(8) if np.array_equal(x[i],codes[j])),-1);assert s>=0;consistent+=np.array_equal(y[i],codes[truth[s]]);assert np.array_equal(a['states'][i],a['states'][i-1])
            assert correct[np.setdiff1d(range(8),changed)].all()
            if arm=='native':assert correct.all() and consistent==16
            epoch_rows.append(dict(epoch=e,changed=int(correct[changed].sum()),retained=6,feedback_consistent=int(consistent),feedback_total=16))
        if r['mode']=='native':
            out=run['stdout'];assert r['device_free'] and out['program_calls']==1 and out['learning_parameter_resets']==out['weight_restore_calls']==0 and out['host_state_seeds']==1 and out['cleanup']=='unprogrammed';assert np.array_equal(get('initial.ram'),initial)
            for tag in ['initial','final']:assert np.array_equal(get('fixed.'+tag),fg(arm+'.fixed'))
            desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');assert np.array_equal(desc[:,7],addr) and np.array_equal(desc[:,1],inp)
            for i,e in enumerate(events):
                assert ((int(desc[i,8])>>16)&1023)==(e['label']+1 if e['label']>=0 else 0)
                assert 0x20000100<=inp[i] and int(inp[i])+256<=0x20100000
                if e['source']>=0:
                    src=e['source'];assert inp[i]==addr[src]+32 and i-src<64;assert all(addr[j]!=addr[src] for j in range(src+1,i+1))
                assert np.array_equal(get('packet'+str(i)).view('u1')[32:],y[i])
            quiet=get('quiet').reshape(epochs,9);assert np.all(quiet[:,0]==np.arange(epochs)) and np.all(quiet[:,3]-quiet[:,2]>=5_000_000) and np.all(quiet[:,5]==0) and np.all(quiet[:,6]==1) and np.all(quiet[:,8]>>16==quiet[:,7]);assert sum(e['source']<0 for e in events if e['kind'] in ['pre_prediction','feedback'])==1
            if seed==SEEDS[0] and arm=='native' and epochs==2:
                pos=next(e['index'] for e in events if e['epoch']==1 and e['kind']=='fresh_example');before=states[np.flatnonzero(ids==pos-1)[0]];after=states[np.flatnonzero(ids==pos)[0]];
                if (q,full) in selector:assert np.array_equal(before,selector[(q,full)][0]) and np.array_equal(after,selector[(q,full)][1]),'exact selector-control repeat'
                selector[(q,full)]=(before.copy(),after.copy(),name,get('scores').copy())
        else:
            ty=get('table_scores').reshape(count,256);tx=get('table_inputs').reshape(count,256);tab=old.copy()
            for i,e in enumerate(events):
                xx=ty[e['source']] if e['source']>=0 else a['inputs'][i];assert np.array_equal(tx[i],xx);scores=codes.astype('i4')@xx.astype('i4');context=int(np.argmax(scores));assert scores[context]>=28;assert np.array_equal(ty[i],codes[tab[context]])
                if e['label']>=0:tab[context]=e['label']
                if arm=='native' and e['kind']=='query':truth=new if e['epoch']%2==0 else old;assert np.array_equal(ty[i],codes[truth[e['context']]])
            if arm=='native' and epochs==12 and not q:
                o=run['stdout'];den=o['repeats']*count;timings.append(dict(seed=seed,exact_cpu_ns_per_event=o['exact_wall_ns']/den,task_table_ns_per_event=o['table_wall_ns']/den))
        rows.append(dict(name=name,mode=r['mode'],seed=seed,arm=arm,epochs=epochs,extra_queries=q,full_queries=full,events=count,changed=sum(v['changed'] for v in epoch_rows),retained=sum(v['retained'] for v in epoch_rows),feedback_consistent=sum(v['feedback_consistent'] for v in epoch_rows),feedback_total=16*epochs,cumulative_connection_replacements=replacements,updating_events=update_events,captured_boundaries=len(ids),rows=epoch_rows))
      assert set(selector)=={(0,False),(1,False),(8,False),(8,True)};base=selector[(0,False)];selector_rows=[]
      assert np.array_equal(selector[(8,False)][3].reshape(-1,256)[46:54],selector[(8,True)][3].reshape(-1,256)[46:54]),'same query outputs at matched descriptor count'
      for (q,full),(b,af,n,_) in sorted(selector.items()):
        assert np.array_equal(b,base[0]);bits=sum(int(v^w).bit_count() for v,w in zip(af,base[1]));assert bits==(6 if q==8 and not full else 0);selector_rows.append(dict(extra_queries=q,full_queries=full,visible_ram_before_identical=True,next_partial_edit_differing_bits=bits,run=n))
    physical=[r for r in rows if r['mode']=='native'];primary=[r for r in physical if r['epochs']==12 and not r['extra_queries']];assert len(physical)==29 and len(primary)==20 and len(failures)==1 and len(superseded)==25
    result=dict(status='INDEPENDENT_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],valid_physical_sessions=len(physical),failed_physical_sessions=len(failures),verified_forwards=sum(r['events'] for r in physical),preserved_failed_prefix_forwards=sum(r['observed_completed_forwards'] for r in failures),primary_seeds=5,changes_per_resident_run=12,primary={arm:dict(changed_recalled=sum(r['changed'] for r in primary if r['arm']==arm),changed_total=120,retained=sum(r['retained'] for r in primary if r['arm']==arm),retained_total=360,feedback_consistent=sum(r['feedback_consistent'] for r in primary if r['arm']==arm),feedback_total=960,cumulative_connection_replacements=sum(r['cumulative_connection_replacements'] for r in primary if r['arm']==arm)) for arm in ARMS},selector_control=selector_rows,failed_attempts=failures,superseded_cpu_forecasts=superseded,cpu_timings=timings,rows=rows,scope='host-paced observation/update/feedback in one resident graph; external synthetic labels; two alternating maps; no indefinite autonomy, electrical spike tracing, CPU or energy advantage')
    save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k not in ['rows','superseded_cpu_forecasts']}))
