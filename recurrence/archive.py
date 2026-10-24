"""Portable exact-state audit. NumPy only; no vendor SDK or physical device."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-native-feedback';BASE=P/'results/20261024-native-feedback'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def masks(a):
    return np.unpackbits(np.asarray(a,dtype='<u4')[...,2:].reshape(*a.shape[:-1],128,8)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def replay(ram,seed,probes,arm,h):
    c=Reference(ram,n=128,u=128,nw=32,classes=1);x=seed.copy();ys=[];states=[];inputs=[]
    def step(a):
        if arm=='frozen':
            bits=sum(1<<int(k) for k in np.flatnonzero(a));p=np.array([(w&bits).bit_count() for w in c.weights])
        else:p=np.array(c.forward(a,0))
        return (p>13).astype('u1')
    for i in range(h):
        inputs.append(x);y=step(x);ys.append(y);states.append(c.ram[:]);x=seed.copy() if arm=='repeated_seed' else np.zeros(128,'u1') if arm=='severed' else y.copy()
    py=[];ps=[]
    for a in probes:py.append(step(a));ps.append(c.ram[:])
    return np.asarray(ys),np.asarray(states,'<u4'),np.asarray(py),np.asarray(ps,'<u4'),np.asarray(inputs)
def build():
    arrays={};fixtures={};runs={}
    for d in sorted((S/'fixtures01').glob('s*')):
        m=json.loads((d/'BUILD.json').read_text());assert m['status']=='COMPILER_AND_REFERENCE_PASS';fs={}
        for name,t in [('initial.ram','<u4'),('fixed.ram','<u4'),('seed.input','u1'),('probes.inputs','u1')]:
            p=d/name;key=d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
        fixtures[d.name]=dict(metadata=m,arrays=fs)
    for d in sorted((S/'raw').iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0;fs={}
        for p in d.glob('result.*'):
            ext=p.name[7:];t='u1' if ext in ['scores','probe_scores'] else '<i8' if ext in ['quiet','idle'] else '<i4' if ext=='potentials' else '<u4';key=d.name+'__'+ext.replace('.','_');arrays[key]=np.fromfile(p,t);fs[ext]=dict(key=key,sha256=sha(p))
        runs[d.name]=dict(receipt=r,arrays=fs,stdout=json.loads((d/'stdout').read_text()),stderr=(d/'stderr').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,source_freeze=json.loads((S/'evidence/FROZEN01.json').read_text()),build_diagnostic=(S/'build01.stderr').read_text(),scope='Bounded pre-scheduled native output-to-input SRAM feedback; no indefinite scheduling or autonomous environmental knowledge'))
    return audit()
def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for block in [*m['fixtures'].values(),*m['runs'].values()]:
            for a in block['arrays'].values():assert hashlib.sha256(z[a['key']].tobytes()).hexdigest()==a['sha256']
        for name,run in m['runs'].items():
            r=run['receipt'];h=r['horizon'];arm=r['arm'];f=m['fixtures'][f's{r["seed"]}'];get=lambda k:z[run['arrays'][k]['key']];fg=lambda k:z[f['arrays'][k]['key']]
            initial=fg('initial.ram');seed=fg('seed.input');probes=fg('probes.inputs').reshape(4,128)
            y,st,py,ps,inputs=replay(initial,seed,probes,arm,h)
            assert np.array_equal(get('scores').reshape(h,128),y);assert np.array_equal(get('ram'),st[-1]);assert np.array_equal(get('probe_scores').reshape(4,128),py);assert np.array_equal(get('probe_ram'),ps[-1]);assert np.all(masks(st).sum(-1)==32)
            assert np.array_equal(st[-1],ps[-1]),'post-loop partial-cue probes do not edit memory'
            initial_masks=masks(initial);terminal=masks(st[-1]);added=int(((terminal==1)&(initial_masks==0)).sum());removed=int(((terminal==0)&(initial_masks==1)).sum());assert added==removed
            actual=dict(name=name,mode=r['mode'],seed=r['seed'],arm=arm,horizon=h,connection_replacements=added,partial_cue_support=py.sum(1).tolist())
            if r['mode']=='native':
                assert r['device_free'] and run['stdout']['cleanup']=='unprogrammed'
                desc=get('descriptors').reshape(h,16);outs=get('output_addresses');assert np.array_equal(outs,desc[:,7]);assert np.all(desc[:,1]>=0x20000100) and np.all(desc[:,1].astype('u8')+128<=0x20100000)
                assert len(np.unique(outs))==h,'distinct output buffers preserve all dependent results'
                for i in range(h):
                    p=get('output_packet'+str(i)).view('u1');assert np.array_equal(p[32:160],y[i])
                    if i and (arm in ['native','frozen'] or arm=='severed' and i>1):assert desc[i,1]==outs[i-1]+32
                    elif i and arm=='repeated_seed':assert desc[i,1]==desc[0,1]
                q,idle=get('quiet'),get('idle');assert q[3]==0 and q[4]==1 and q[6]>>16==q[5] and q[1]-q[0]>=5_000_000
                assert idle[3]==idle[4]==0 and idle[5]==q[6] and idle[1]-idle[0]>=20_000_000
                assert np.array_equal(get('idle.ram'),st[-1]);assert np.array_equal(get('initial.ram'),initial)
                for tag in ['initial','final']:assert np.array_equal(get('fixed.'+tag),fg('fixed.ram'))
                actual.update(quiet_wall_ns=int(q[1]-q[0]),quiet_worker_cpu_ns=int(q[2]),idle_wall_ns=int(idle[1]-idle[0]))
            if h==32:
                assert py.sum(1).tolist()==([1]*4 if arm=='native' else [0]*4 if arm=='frozen' else [int(g==f['metadata']['seed_state']) for g in range(4)])
                if arm=='native':assert added==64
                elif arm=='frozen':assert added==0
                else:assert added==16
            rows.append(actual)
        # Prefix snapshots are independently observed replicas, not invented
        # per-event readbacks during the 32-step quiet execution.
        for h in [1,2,3,4,5,8,16,32]:
            n=f'native_s109101_native_h{h}';assert n in m['runs']
            run=m['runs'][n];r=run['receipt'];f=m['fixtures']['s109101'];replayed=replay(z[f['arrays']['initial.ram']['key']],z[f['arrays']['seed.input']['key']],z[f['arrays']['probes.inputs']['key']].reshape(4,128),'native',32)
            assert np.array_equal(z[run['arrays']['ram']['key']],replayed[1][h-1])
    native=[r for r in rows if r['mode']=='native'];assert len(native)==22 and len(rows)==44
    result=dict(status='INDEPENDENT_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=len(native),physical_forwards=sum(r['horizon']+4 for r in native),primary_32_step_seeds=3,primary_partial_cue_hits=sum(sum(r['partial_cue_support']) for r in native if r['arm']=='native' and r['horizon']==32),frozen_partial_cue_hits=0,repeated_seed_partial_cue_hits=3,severed_partial_cue_hits=3,quiet_bound='complete by first wake after 5 ms; not a device latency measurement',post_exhaustion_bound='completed-job status and learning SRAM unchanged over subsequent 20 ms; raw output payloads not reread after that idle interval',rows=rows)
    save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
