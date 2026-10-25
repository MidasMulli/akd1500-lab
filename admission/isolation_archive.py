"""Portable numeric evidence for native co-resident correction and isolation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from admission import isolation_run as mod
from admission.run import save,sha
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-isolated-correction'
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def dtype(name):return 'u1' if name in ['inputs','scores'] else '<i4' if name.startswith('potentials') or name=='learning_modes' else '<u4'

def build():
    arrays={};fixtures={};runs={}
    for f in sorted(mod.F.glob('s*')):
        ff={}
        for p in sorted([*f.glob('*.ram'),*f.glob('*.threshold')]):
            key=f'{f.name}__{p.name.replace(".","_")}';arrays[key]=np.fromfile(p,'<u4');ff[p.name]=dict(key=key,sha256=sha(p))
        fixtures[f.name]=dict(arrays=ff,metadata=json.loads((f/'BUILD.json').read_text()))
    for d in sorted((mod.S/'raw').iterdir()):
        if not (d/'RECEIPT.json').exists():continue
        ff={};packets=[]
        for p in sorted(d.glob('result.*')):
            name=p.name[7:]
            if name.startswith('packet'):packets.append((int(name[6:]),np.fromfile(p,'<u4')));continue
            key=f'{d.name}__{name.replace(".","_")}';arrays[key]=np.fromfile(p,dtype(name));ff[name]=dict(key=key,sha256=sha(p))
        packet_meta=None
        if packets:
            packets.sort(key=lambda q:q[0]);assert [i for i,_ in packets]==list(range(len(packets)))
            key=f'{d.name}__packets';arrays[key]=np.stack([v for _,v in packets]);packet_meta=dict(key=key,sha256=digest(arrays[key]),rows=len(packets))
        runs[d.name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),
            audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,
            arrays=ff,packets=packet_meta,stdout=(d/'stdout').read_text(),stderr=(d/'stderr').read_text(),source_freeze=json.loads((d/'FROZEN.json').read_text()))
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,
        closed_state=json.loads((mod.S/'STATE.json').read_text()),compiler_attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in mod.S.glob('fixtures*/*/BUILD.json') if json.loads(p.read_text())['status']!='VIRTUAL_COMPILER_PASS'],
        scope='Two context-partitioned correcting banks (16 plastic, 80 fixed helper records), seven sequential FNP3s. Host raw contexts/cues/outcomes and fixed query/positive/negative clocks; native routing, error comparison and connection updates. No host context-dependent bank clock, record selection, weight repair or reset in native arm. Engineered partition/coding/rule; CPU also passes; no energy, speed or autonomous scheduler claim.',
        provenance=dict(measured='native input/output packets, both complete RAM trajectories, initial/final fixed inference RAM, native descriptor/source addresses',
            derived='intermediate features, packed potentials, dictionary/LRU CPU controls',journal='learning_modes and host counters are own source/instrument journals, not electrical CSR/bus activity traces')))
    return audit()

def get_run(z,m,name):
    r=m['runs'][name];f=m['fixtures'][f"s{r['receipt']['seed']}_{r['receipt']['arm']}"]
    def get(k):
        if k.startswith('packet'):return z[r['packets']['key']][int(k[6:])]
        return z[r['arrays'][k]['key']]
    return r,get,lambda k:z[f['arrays'][k]['key']]

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];failed=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for f in m['fixtures'].values():
            for p in f['arrays'].values():assert digest(z[p['key']])==p['sha256']
        for name,run in m['runs'].items():
            for p in run['arrays'].values():assert digest(z[p['key']])==p['sha256']
            if run['packets']:
                p=run['packets'];assert digest(z[p['key']])==p['sha256']
                for i,row in enumerate(z[p['key']]):assert digest(row)==run['receipt']['hashes'][f'result.packet{i}']
            if not run['verified']:failed.append(dict(name=name,receipt=run['receipt'],stderr=run['stderr']));continue
            r,get,fg=get_run(z,m,name);receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']));a=mod.check(receipt,get,fg);a['name']=name;rows.append(a)
        assert m['closed_state']['closed'] and not m['closed_state'].get('failure')
        native={(r['seed'],r['arm'],r['stage'],r['stream']):r for r in rows if r['mode']=='native'}
        cells=[(s,mod.STREAMS[0]) for s in mod.SEEDS]+[(mod.SEEDS[0],w) for w in mod.STREAMS[1:]]
        required={(mod.SEEDS[0],arm,'truth',mod.STREAMS[0]) for arm in ['native','open','closed']}|{(mod.SEEDS[0],arm,'gate',mod.STREAMS[0]) for arm in ['native','no_retire','frozen']}
        required|={(s,arm,'repeat',w) for s,w in cells for arm in ['native','no_retire','frozen']}
        required|={(mod.SEEDS[0],arm,'repeat',mod.STREAMS[0]) for arm in ['solo_A','solo_B','freeze_A','freeze_B']}
        required|={(mod.SEEDS[0],arm,'pressure',mod.STREAMS[0]) for arm in ['native','no_retire','frozen']}
        assert set(native)==required and len(native)==31 and sum(r['events'] for r in native.values())==31500,'complete frozen physical series'
        primary={}
        for arm in ['native','no_retire','frozen']:
            rs=[native[s,arm,'repeat',w] for s,w in cells];probes=[v for r in rs for k,v in r['assessment'].items() if k.startswith('probe_')];allq=[v for r in rs for v in r['assessment'].values()]
            primary[arm]=dict(correct=sum(v['correct'] for v in probes),total=sum(v['queries'] for v in probes),all_correct=sum(v['correct'] for v in allq),all_total=sum(v['queries'] for v in allq),connection_replacements=sum(e['connection_replacements'] for r in rs for e in r['edits']),foreign_boundaries_unchanged=sum(e['foreign_boundaries_unchanged'] for r in rs for e in r['edits']),stable_records=sum(len(r['stable_records']) for r in rs))
        rs=[native[s,'native','repeat',w] for s,w in cells];probes=[v for r in rs for k,v in r['assessment'].items() if k.startswith('probe_')];allq=[v for r in rs for v in r['assessment'].values()]
        cpu={name:dict(correct=sum(v[key] for v in probes),total=6912,all_correct=sum(v[key] for v in allq),all_total=sum(v['queries'] for v in allq)) for name,key in [('dictionary','cpu'),('lru8_per_context','cpu_lru8'),('context_ablated_dictionary','aliased_cpu')]}
        solo={};seed=mod.SEEDS[0];stream=mod.STREAMS[0];mixed_name=f'native_s{seed}_native_repeat_w{stream}'
        mixed,rget,fg=get_run(z,m,mixed_name);a=mod.simulate([fg(f'bank{k}.ram') for k in range(2)],'repeat','native',stream)
        for ctx,arm in enumerate(['solo_A','solo_B']):
            _,get,_=get_run(z,m,f'native_s{seed}_{arm}_repeat_w{stream}');ii=[e['index'] for e in a['events'] if e['context']==ctx]
            x=rget(f'states{ctx}').reshape(len(a['events']),-1)[ii];y=get(f'states{ctx}').reshape(len(ii),-1)
            solo[arm]=dict(matched_context_boundaries=len(ii),identical_visible_trajectory=bool(np.array_equal(x,y)),different_words=int((x!=y).sum()),identical_final_ram=bool(np.array_equal(x[-1],y[-1])))
        pressure={arm:native[seed,arm,'pressure',stream]['assessment']['capacity_probe'] for arm in ['native','no_retire','frozen']}
        frozen_banks={arm:{str(ctx):sum(v['by_context'][str(ctx)]['correct'] for k,v in native[seed,arm,'repeat',stream]['assessment'].items() if k.startswith('probe_')) for ctx in range(2)} for arm in ['freeze_A','freeze_B']}
    out=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=31,native_forwards=31500,primary=primary,cpu=cpu,solo=solo,frozen_banks=frozen_banks,pressure=pressure,design_cells=[dict(seed=s,stream=w) for s,w in cells],preserved_execution_failures=failed,rows=rows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
