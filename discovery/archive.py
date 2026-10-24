"""Portable numeric-only archive and source-derived replay; no SDK dependency."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from discovery.model import *
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-context-discovery';BASE=P/'results/20261024-context-discovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def masks(v):return np.unpackbits(np.asarray(v,dtype='<u4')[...,2:].reshape(*v.shape[:-1],U,12)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def build():
    arrays={};fixtures={};runs={}
    for d in sorted((S/'fixtures05').glob('s*')):
        fs={}
        for p in [d/'initial.ram',*sorted(d.glob('fixed*.ram'))]:
            key=d.name+'__'+p.name.replace('.','_');arrays[key]=np.fromfile(p,'<u4');fs[p.name]=dict(key=key,sha256=sha(p))
        fixtures[d.name]=dict(arrays=fs,metadata=json.loads((d/'BUILD.json').read_text()))
    for d in sorted((S/'raw').iterdir()):
        fs={}
        for p in sorted(d.glob('result.*')):
            name=p.name[7:];t='u1' if name in ['inputs','scores'] else '<i4' if name in ['cpu','potentials','learning_modes'] else '<u4';key=d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
        raw=(d/'stdout').read_text();runs[d.name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),superseded_reference=d.name in ['cpu_s113101_native_c1_attempt2','cpu_s113101_native_c1_attempt3'],audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,arrays=fs,stdout=json.loads(raw) if raw.startswith('{') else raw,stderr=(d/'stderr').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    compiler_failures=[json.loads(p.read_text()) for p in sorted(S.glob('fixtures0[12]/*/BUILD.json'))]
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,compiler_failures=compiler_failures,field_provenance=dict(native_measured='inputs,scores,states,ram,initial.ram,descriptors,input/output addresses,packet*,fixed*.initial/final',derived='potentials: packed CPU predictions; helper and feature activity from fixed weights + captured inputs',host_journal='learning_modes is requested global mode, not a direct register trace; receipt counters and cleanup are instrument logs'),source_freezes={p.name:json.loads(p.read_text()) for p in sorted((S/'evidence').glob('FROZEN*.json'))},scope='Measured native input/output packets and per-event learning RAM. Fixed three-bit conjunction basis and fixed history helpers; single constant training label0,32 competing plastic records; host-paced events and explicit global query learning disable. No outcome prediction or CPU efficiency advantage tested.'))
    return audit()
def check(run,z,fixture):
    r=run['receipt'];get=lambda k:z[run['arrays'][k]['key']];fg=lambda k:z[fixture['arrays'][k]['key']]
    seed,arm,cycles=r['seed'],r['arm'],r['cycles'];initial=fg('initial.ram');assert np.array_equal(masks(initial).T,weights(seed))
    a=simulate(initial,cycles,arm);events=a['events'];count=len(events);x=get('inputs').reshape(count,N);y=get('scores').reshape(count,N);states=get('states').reshape(count,482)
    assert np.array_equal(x,a['inputs']),'native carried input semantics'
    assert np.array_equal(y,a['outputs']),'returned context activity'
    assert np.array_equal(states,a['states']) and np.array_equal(get('ram'),states[-1]),'learning SRAM replay'
    assert np.array_equal(get('potentials').reshape(count,U),a['potentials']) and np.array_equal(get('cpu'),a['cpu'])
    assert np.all(masks(states).sum(-1)==32)
    for j in range(LEARNED,U):assert np.all(states[:,2+j*12:2+(j+1)*12]==initial[2+j*12:2+(j+1)*12]),'fixed helpers'
    for i,e in enumerate(events):
        if e['label']<0:assert np.array_equal(states[i],states[i-1] if i else initial),'read-only query'
    old={}
    if arm!='unlabelled':
        for e in events:
            if e['phase']=='old_probe':
                ids=np.flatnonzero(y[e['index'],64:96]);assert len(ids)==1;old[e['history']]=int(ids[0])
        assert len(old)==len(set(old.values()))==4
    result=assess(dict(a,outputs=y,states=states))
    if arm in ['native','unseen'] and cycles==12:
        assert result['histories']==result['unique_records']==16 and result['ambiguous']==result['missing']==0
        assert all(result['assignments'][k]==j for k,j in old.items())
        for e in events:
            if e['phase']=='probe':assert y[e['index'],64:96].sum()==1 and y[e['index'],64+result['assignments'][e['history']]]==1
        assert len({int(get('cpu')[e['index']]) for e in events if e['phase']=='probe'})==16
    if arm in ['frozen','unseen_frozen']:assert result['assignments']==old and result['missing']==12
    if arm in ['unlabelled','severed']:assert not y[[e['index'] for e in events if e['phase']=='probe'],64:96].any()
    before=masks(initial);replacements=0
    for state in states:
        after=masks(state);replacements+=int(((after==1)&(before==0)).sum());before=after
    result.update(name=run.get('name',''),mode=r['mode'],seed=seed,arm=arm,cycles=cycles,events=count,replacements=replacements,old_assignments=old)
    if arm in ['unseen','unseen_frozen']:
        held=[e for e in events if e['phase']=='heldout'];assert len(held)==128 and len({e['history'] for e in held})==16
        for e in held:
            j=result['assignments'].get(e['history']);v=y[e['index'],64:96]
            assert (v.sum()==1 and v[j]==1) if j is not None else not v.any()
        result.update(heldout_responses=sum(int(y[e['index'],64:96].sum()==1) for e in held),heldout_total=128)
        if arm=='unseen':assert np.all(get('cpu')[[e['index'] for e in held]]>=0)
    if r['mode']=='native':
        o=run['stdout'];assert r['exit_code']==0 and r['device_free'] and o['program_calls']==1 and o['weight_restore_calls']==o['threshold_resets']==0 and o['cleanup']=='unprogrammed' and o['current_symbol_writes']==count
        assert o['history_ablation_writes']==sum(e['sever'] for e in events)
        assert np.array_equal(get('initial.ram'),initial)
        for j in range(2):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{j}.'+tag),fg(f'fixed{j}.ram'))
        assert get('learning_modes').tolist()==[int(e['label']>=0) for e in events]
        desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        for i,e in enumerate(events):
            assert ((int(desc[i,8])>>16)&1023)==(1 if e['label']==0 else 0),'constant label only'
            assert 0x20000100<=inp[i] and int(inp[i])+N<=0x20100000
            if i:
                assert inp[i]==addr[i-1]+32 and addr[i]!=addr[i-1]
                if e['sever']:assert not x[i,:48].any()
                else:assert np.array_equal(x[i,:48],y[i-1,:48]),'native carried history'
                assert np.array_equal(x[i,64:],y[i-1,64:])
            assert np.array_equal(get('packet'+str(i)).view('u1')[32:],y[i])
    return result

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];failures=[];superseded=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for b in [*m['fixtures'].values(),*m['runs'].values()]:
            for q in b['arrays'].values():assert digest(z[q['key']])==q['sha256']
        for name,run in m['runs'].items():
            run['name']=name
            if run.get('superseded_reference'):
                assert run['receipt']['mode']=='cpu';superseded.append(name);continue
            if not run['verified']:
                failures.append(dict(name=name,exit_code=run['receipt']['exit_code'],stderr=run['stderr'],captured_forwards=len(z[run['arrays']['scores']['key']])//N if 'scores' in run['arrays'] else 0));continue
            rows.append(check(run,z,m['fixtures'][f"s{run['receipt']['seed']}"]))
    physical=[r for r in rows if r['mode']=='native'];main={(r['seed'],r['arm']):r for r in physical if r['cycles']==12}
    required={(s,a) for s in SEEDS for a in ['native','frozen','unlabelled']}|{(SEEDS[0],'severed')}|{(s,'unseen') for s in SEEDS}|{(SEEDS[0],'unseen_frozen')};assert set(main)==required
    assert len(physical)==25 and sum(r['events'] for r in physical)==8379
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for seed in SEEDS:
            def arr(arm,key):return z[m['runs'][f'native_s{seed}_{arm}_c12']['arrays'][key]['key']]
            assert np.array_equal(arr('native','inputs').reshape(-1,N)[:,:64],arr('frozen','inputs').reshape(-1,N)[:,:64]),'matched raw symbol/history streams'
            assert np.array_equal(arr('native','states'),arr('unseen','states')[:323*482]),'heldout run prefix unchanged'
            assert np.array_equal(arr('native','scores'),arr('unseen','scores')[:323*N])
        assert np.array_equal(z[m['runs']['native_s113101_native_c12']['arrays']['states']['key']],z[m['runs']['native_s113101_severed_c12']['arrays']['states']['key']]),'history ablation same learning RAM'
    result=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=len(physical),verified_forwards=sum(r['events'] for r in physical),failed_attempts=failures,superseded_cpu_reference_runs=superseded,heldout_native=dict(correct=sum(main[(s,'unseen')]['heldout_responses'] for s in SEEDS),total=640),heldout_trained_frozen=dict(responses=main[(SEEDS[0],'unseen_frozen')]['heldout_responses'],total=128),compiler_failures=m['compiler_failures'],primary={arm:dict(distinct_histories=sum(main[(s,arm)]['unique_records'] for s in SEEDS),total=80) for arm in ['native','frozen','unlabelled']},old_context_id_retention=dict(retained=20,total=20),cpu_dictionary=dict(histories=80,total=80),rows=rows,scope=m['scope']);save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k not in ['rows','compiler_failures']}))
