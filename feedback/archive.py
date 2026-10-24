"""Portable read-only replay of input-gated learning and native readout."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from feedback.design import P,S,simulate,metrics
BASE=P/'results/20261024-predictive-feedback'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def build():
    arrays={};fixtures={};runs={};failures=[]
    for d in sorted((S/'fixtures02').glob('s*')):
        r=json.loads((d/'BUILD.json').read_text());assert r['status']=='COMPILER_AND_CPU_SCREEN_PASS';fs={}
        for p in d.iterdir():
            if p.suffix in ['.bin','.json']:continue
            if p.suffix=='.layout':continue
            typ='u1' if p.suffix=='.inputs' else '<i4' if p.suffix=='.potentials' else '<u4';key=d.name+'__'+p.name.replace('.','_');arrays[key]=np.fromfile(p,typ);fs[p.name]=dict(key=key,sha256=sha(p))
        fixtures[d.name]=dict(metadata=r,arrays=fs,rows={a:json.loads((d/(a+'.rows.json')).read_text()) for a in ['native','frozen','disconnected']})
    for d in sorted((S/'raw').iterdir()):
        if (d/'ADMISSION_FAILURE.json').exists():failures.append(json.loads((d/'ADMISSION_FAILURE.json').read_text()));continue
        rec=json.loads((d/'RECEIPT.json').read_text());aud=json.loads((d/'AUDIT.json').read_text());assert rec['exit_code']==0 and aud['status']=='SAVED_ARRAY_PASS';fs={}
        for p in d.glob('result.*'):
            ext=p.name[7:];typ='<i8' if ext.endswith('_ns') or ext=='blocked_intervals' else '<i4' if ext in ['scores','boundaries'] else '<u4';key=d.name+'__'+ext.replace('.','_');arrays[key]=np.fromfile(p,typ);fs[ext]=dict(key=key,sha256=sha(p))
        runs[d.name]=dict(receipt=rec,audit=aud,arrays=fs)
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,admission_failures=failures,compiler_failures=[json.loads(p.read_text()) for p in S.glob('fixtures01/*/BUILD.json')],source_freeze=json.loads((S/'evidence/FROZEN.json').read_text()),scope='Input-gated native plasticity; external observation encoder and bounded host queues; not indefinite recurrence'))
    return audit()
def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];results=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for r in m['fixtures'].values():
            for a in r['arrays'].values():assert hashlib.sha256(z[a['key']].tobytes()).hexdigest()==a['sha256']
        for name,run in m['runs'].items():
            r=run['receipt'];f=m['fixtures'][f's{r["seed"]}'];get=lambda k:z[run['arrays'][k]['key']];fg=lambda k:z[f['arrays'][k]['key']]
            for a in run['arrays'].values():assert hashlib.sha256(z[a['key']].tobytes()).hexdigest()==a['sha256']
            rows=f['rows'][r['arm']][:r['count']];x=fg(r['arm']+'.inputs').reshape(768,512)[:r['count']];ram=fg(r['variant']+'.ram')
            potentials,states=simulate(ram,x,rows,r['arm']);before=np.vstack([ram,states[:-1]])
            if r['mode']!='task':
                expected=potentials if r['variant']=='raw' else (potentials>13).sum(axis=1,keepdims=True);actual=get('scores').reshape(expected.shape);assert np.array_equal(actual,expected)
                assert np.array_equal(get('snapshots').reshape(-1,322),states[get('boundaries')]);assert np.array_equal(get('ram'),states[-1])
            else:actual=get('scores')
            for i,row in enumerate(rows):
                if not row['gate_present'] or r['arm']=='frozen' and i>=128:assert np.array_equal(before[i],states[i])
                # Anchored context and feedback connections are never removed.
                weights=states[i,2:].reshape(16,20)[:,4:].copy();bits=np.unpackbits(weights.view('u1'),axis=1,bitorder='little')
                assert np.all(bits.sum(1)==32)
                for j in range(16):assert bits[j,j*10:j*10+10].sum()==10 and bits[j,160:176].sum()==16
            if r['mode']=='native':
                for key in f['arrays']:
                    if key.startswith(r['variant']+'.packet') and key.endswith('.ram'):
                        packet=key.split('.')[1]
                        for tag in ['initial','final']:assert np.array_equal(get(packet+'.'+tag),fg(key))
            # RAM only changes on actual outcome-feedback events. Count actual
            # connection replacements separately from metadata/teaching count.
            a=before[:,2:].reshape(-1,16,20)[:,:,4:].copy();b=states[:,2:].reshape(-1,16,20)[:,:,4:].copy();edits=np.any(a!=b,axis=(1,2))
            v=metrics(actual if r['variant']=='raw' and r['mode']!='task' else actual.reshape(-1),rows,f['metadata']['truth'])['groups'];assert v==run['audit']['metrics']
            results.append(dict(name=name,mode=r['mode'],seed=r['seed'],arm=r['arm'],variant=r['variant'],count=r['count'],batch=r['batch'],actual_connection_edit_events=int(edits.sum()),metrics=v))
    result=dict(status='INDEPENDENT_ARCHIVE_PASS',physical_sessions=sum(r['mode']=='native' for r in results),physical_forwards=sum(r['count'] for r in results if r['mode']=='native'),runs=len(results),preserved_nonexecuted_admissions=len(m['admission_failures']),compiler_failures=len(m['compiler_failures']),archive_sha256=m['archive_sha256'],rows=results);save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
