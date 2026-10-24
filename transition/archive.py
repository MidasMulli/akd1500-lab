"""Portable raw archive and independent scalar replay. NumPy, no SDK or device."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-learned-transition';BASE=P/'results/20261024-learned-transition'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def masks(a):return np.unpackbits(np.asarray(a,dtype='<u4')[...,2:].reshape(*a.shape[:-1],8,12)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def build():
    arrays={};fixtures={};runs={};replays={}
    for f in sorted((S/'fixtures01').glob('s*')):
        fs={}
        for name in ['codes','partial','old','new','changed','initial.ram','fixed.ram']:
            p=f/name;t='u1' if name in ['codes','partial'] else '<i4' if name in ['old','new','changed'] else '<u4';key=f.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
        fixtures[f.name]=dict(metadata=json.loads((f/'BUILD.json').read_text()),arrays=fs)
    for folder,dest in [('raw',runs),('replay_raw',replays)]:
      for d in sorted((S/folder).iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text());fs={}
        for p in sorted(d.glob('result.*')):
            name=p.name[7:];t='u1' if name in ['scores','rollout'] or name.startswith('seed') else '<i8' if name in ['quiet','latency_ns'] else '<i4' if name=='potentials' else '<u4';key=folder+'__'+d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
        stdout=(d/'stdout').read_text();dest[d.name]=dict(receipt=r,arrays=fs,stdout=json.loads(stdout) if stdout.strip().startswith('{') else stdout,stderr=(d/'stderr').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    cleanup=S/'evidence/REPLAY_CLEANUP';cr=json.loads((cleanup/'RECEIPT.json').read_text());assert cr['exit_code']==0 and cr['device_free']
    after=np.fromfile(cleanup/'result.after','<u4');assert len(after)==3 and np.all((after&1)==0)
    metadata=dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,replays=replays,source_freezes={p.name:json.loads(p.read_text()) for p in sorted((S/'evidence').glob('*FROZEN*.json'))},repairs=json.loads((S/'evidence/REPAIRS.json').read_text()),compiler_replay_screen=json.loads((S/'replay_fixtures01/BUILD.json').read_text()),replay_adapters={f:json.loads((S/f/'BUILD.json').read_text()) for f in ['replay_fixtures02','replay_fixtures03']},compile_failures=[json.loads(p.read_text()) for p in (S/'failed_attempts').glob('*/FAILURE.json')],cleanup=dict(receipt=cr,dma_control_after=after.tolist()),scope='Externally taught, class-addressed FNP3 transition memory with fixed neural decoder and finite SRAM feedback. Separate two-pass inference replay failed before inference.')
    save(BASE.with_suffix('.json'),metadata);return audit()
def audit():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==meta['archive_sha256'];rows=[];failed=[];replay_rows=[];timings=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
      for block in [*meta['fixtures'].values(),*meta['runs'].values(),*meta['replays'].values()]:
        for a in block['arrays'].values():assert hashlib.sha256(z[a['key']].tobytes()).hexdigest()==a['sha256']
      for name,run in meta['runs'].items():
        r=run['receipt'];get=lambda k:z[run['arrays'][k]['key']]
        if r['exit_code']:
            failed.append(dict(name=name,mode=r['mode'],exit_code=r['exit_code'],error=run['stderr'].strip(),observed_diagnostic_events=int(len(get('states'))//98) if 'states' in run['arrays'] else 0,observed_rollout_outputs=int(len(get('rollout'))//256) if 'rollout' in run['arrays'] else 0));continue
        h=r['horizon'];arm=r['arm'];f=meta['fixtures'][f's{r["seed"]}'];fg=lambda k:z[f['arrays'][k]['key']];codes=fg('codes').reshape(8,256);partial=fg('partial').reshape(8,256);old,new,changed=fg('old'),fg('new'),fg('changed');initial=fg('initial.ram');ref=Reference(initial,n=256,u=8,nw=32,classes=8)
        events=[(codes[s],int(old[s]),True) for s in range(8)]+[(partial[s],-1,False) for s in range(8)]+[(codes[s],int(new[s]),arm=='native') for s in changed]+[(partial[s],-1,False) for s in range(8)]
        yy=[];ss=[]
        def output(x,label,learn):
            bits=sum(1<<int(k) for k in np.flatnonzero(x));p=ref.forward(x,label) if learn else [(w&bits).bit_count() for w in ref.weights];active=np.flatnonzero(np.asarray(p)>24);return np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(256,'u1')
        for x,label,learn in events:yy.append(output(x,label,learn));ss.append(ref.ram[:])
        yy=np.asarray(yy);ss=np.asarray(ss,'<u4');assert np.array_equal(get('scores').reshape(26,256),yy) and np.array_equal(get('states').reshape(26,98),ss);assert np.array_equal(get('ram'),ref.ram);assert np.all(masks(ss).sum(-1)==32)
        assert np.array_equal(ss[7],initial) and np.array_equal(ss[15],initial) and np.all(ss[18:]==ss[17]);edits=int(((masks(ss[-1])==1)&(masks(initial)==0)).sum());assert edits==(32 if arm=='native' else 0)
        post=new if arm=='native' else old;assert np.array_equal(yy[8:16],codes[old]) and np.array_equal(yy[18:26],codes[post])
        actual=get('rollout').reshape(8,h,256);correct=0
        for s in range(8):
            x=partial[s];truth=s
            for t in range(h):
                x=output(x,0,False);assert np.array_equal(x,actual[s,t]);truth=int(new[truth]);correct+=np.array_equal(x,codes[truth])
        assert np.array_equal(ref.ram,ss[-1]),'unlabelled rollout did not change RAM'
        if r['mode']=='native':
            assert r['device_free'] and run['stdout']['cleanup']=='unprogrammed';assert np.array_equal(get('initial.ram'),initial) and np.array_equal(get('rollout.ram'),ss[-1]);assert all(np.array_equal(get('fixed.'+tag),fg('fixed.ram')) for tag in ['initial','final'])
            if h:
                desc=get('descriptors').reshape(8,h,16);outs=get('output_addresses').reshape(8,h);q=get('quiet').reshape(8,7);assert np.array_equal(desc[:,:,7],outs) and np.all((desc[:,:,8]>>16&1023)==0);assert np.all(q[:,1]-q[:,0]>=5_000_000) and np.all(q[:,3]==0) and np.all(q[:,4]==1) and np.all(q[:,6]>>16==q[:,5])
                for s in range(8):
                    assert np.array_equal(get('seed'+str(s)),partial[s]);assert len(np.unique(outs[s]))==h
                    for t in range(h):
                        if t:assert desc[s,t,1]==outs[s,t-1]+32
                        assert 0x20000100<=desc[s,t,1] and int(desc[s,t,1])+256<=0x20100000
                        assert np.array_equal(get('packet'+str(s*h+t)).view('u1')[32:],actual[s,t])
        elif h==32 and arm=='native':
            a=run['stdout'];denom=a['repeats']*(26+8*h);timings.append(dict(seed=r['seed'],exact_cpu_ns_per_event=a['exact_wall_ns']/denom,task_table_ns_per_event=a['table_wall_ns']/denom,contract='includes dense encode/decode and checksum; task CPU differs during transient teaching example, agrees at recall and rollout endpoints'))
        rows.append(dict(name=name,mode=r['mode'],seed=r['seed'],arm=arm,horizon=h,connection_replacements=edits,changed_recall=2 if arm=='native' else 0,retained=6,rollout_correct=int(correct),rollout_total=8*h))
      for name,run in meta['replays'].items():
        r=run['receipt'];get=lambda k:z[run['arrays'][k]['key']]
        if r['exit_code']:replay_rows.append(dict(name=name,status='FAILED_PRESERVED',horizon=r['horizon'],error=run['stderr'].strip(),forward_outputs_observed=0));continue
        assert r['horizon']==1 and r['device_free'];f=meta['fixtures']['s110101'];fg=lambda k:z[f['arrays'][k]['key']];codes=fg('codes').reshape(8,256);assert np.array_equal(get('scores').reshape(8,256),codes[fg('new')]);q=get('quiet').reshape(8,7);assert np.all(q[:,1]-q[:,0]>=5_000_000) and np.all(q[:,3]==0) and np.all(q[:,4]==1) and np.all(q[:,6]>>16==q[:,5]);replay_rows.append(dict(name=name,status='ONE_PASS_BASELINE_PASS',horizon=1,terminal_correct=8))
    native=[r for r in rows if r['mode']=='native'];primary=[r for r in native if r['horizon']==32];assert len(native)==19 and len(primary)==15 and len(failed)==2
    result=dict(status='INDEPENDENT_ARCHIVE_PASS',archive_sha256=meta['archive_sha256'],valid_physical_sessions=len(native),successful_primary_forwards=sum(26+r['rollout_total'] for r in native),invalid_control_forwards_observed=sum(r['observed_diagnostic_events']+r['observed_rollout_outputs'] for r in failed if r['mode']=='native'),primary_seeds=5,changed_recall_native=sum(r['changed_recall'] for r in primary if r['arm']=='native'),changed_recall_total=10,retained_native=30,retained_total=30,rollout={a:dict(correct=sum(r['rollout_correct'] for r in primary if r['arm']==a),total=sum(r['rollout_total'] for r in primary if r['arm']==a)) for a in ['native','frozen','disconnected']},failed_attempts=failed,secondary_replay=replay_rows,cpu_timings=timings,rows=rows,limitations=['engineered prior and externally supplied destination classes; one prototype per class','one changed permutation episode; no unlimited reversal claim','finite prequeued SRAM dependencies, no indefinite self-scheduling','two-pass hardware-generated replay failed during configuration, not inference','visible SRAM readback, not hidden selector or electrical spike tracing','CPU task table also succeeds; no speed or energy advantage established','earlier high-address FNP2 learning bug remains open'])
    save(BASE.with_suffix('.audit.json'),result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
