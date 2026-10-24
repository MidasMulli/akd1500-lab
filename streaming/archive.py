"""Public numeric stream evidence and lossless full-RAM snapshot deltas."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from streaming.design import ROOT,STUDY,N,U,NW,QUERY,encode,observation,band,decode,sha,save,plan
from streaming.audit import audit_run
from pong.relearn.model import Reference,RecentNearest
from resident.or_archive import compact,expand

BASE=ROOT/'results/2026-10-24-streaming-prediction'

def export(study=STUDY):
    state=json.loads((study/'STATE.json').read_text());assert state['research_admissions_closed']
    data={};runs=[]
    for d in sorted((study/'raw').iterdir()):
        result=audit_run(study,d.name);owner=json.loads((d/'OWNER.json').read_text())
        rows=json.loads((d/'rows.json').read_text());events=[json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()]
        replies=[json.loads(s) for s in (d/'native.stdout').read_text().splitlines()];replies=[r for r in replies if r['kind']=='forward']
        initial=np.fromfile(d/'fixture.ram','<u4');states=np.stack([np.fromfile(d/f'native.snapshot_{i}.ram','<u4') for i in range(len(replies))])
        z=dict(initial=initial,scores=np.fromfile(d/'native.scores','<i4').reshape(-1,U),
               inputs=np.fromfile(d/'fixture.inputs',np.uint8).reshape(-1,N),labels=np.fromfile(d/'fixture.labels','<i4'))
        for k,v in compact(initial,states).items():z['delta.'+k]=v
        frames=np.load(d/'frames.npz')
        for k in frames.files:z['frames.'+k]=frames[k]
        clocks=[]
        for i in range(len(rows)):
            q,o,f=events[3*i:3*i+3]
            clocks.append([q['native']['host_monotonic_ns'],q['clock_ns'],o['clock_ns'],f['native']['host_monotonic_ns'],f['clock_ns']])
        z['clock_ns']=np.asarray(clocks,'<i8')
        z['timing_ns']=np.array([[r[k] for k in ['transaction_ns','native_thread_cpu_ns','cpu_reference_ns','cpu_task_ns']] for r in replies],'<i8')
        for key,v in z.items():data[d.name+'__'+key]=v
        runs.append(dict(name=d.name,arm=owner['arm'],seed=owner['seed'],samples=len(rows),native_forwards=len(replies),
                         rows=rows,audit=result,model=json.loads((d/'MODEL_RECEIPT.json').read_text()),
                         learning_np=json.loads((d/'fixture.json').read_text())['learning_np'],
                         query_and_teacher_inputs_equal=True,device_released=owner['device_released']))
    meta=dict(status='CLOSED_STREAMING_PREDICTION_WITH_GATE_DISPOSITION',runs=runs,
              protocol=json.loads((study/'PROTOCOL.json').read_text()),
              pair=json.loads((study/'evidence/PAIR_s105101.json').read_text()),
              model_probe=json.loads((study/'evidence/model_probe/RESULT.json').read_text()),
              executed_source_hashes=json.loads((study/'evidence/SOURCE_FREEZE.json').read_text())['hashes'],
              executed_source_paths=json.loads((study/'evidence/EXECUTED_SOURCE_PATHS.json').read_text()),
              repairs=json.loads((study/'evidence/REPAIRS.json').read_text()),
              admitted_seeds=[105101],unrun_seeds=[105102,105103],skipped_inferences=0,
              limitations=['One chip, one pilot initialization; no confirmation/generalization claim.',
                           'Simulator RGB stream and disclosed pixel tracker; no camera/ESP sensor and no raw-pixel AKD learning.',
                           'One learning FNP with160 allocated records, protected query class; not the whole chip.',
                           'Disclosed actuator-command reversal, not a changed physical world law.',
                           'Frozen T202 checkpoint; not a claim about the latest/best world model.',
                           'Bounded band target does not replace full RGB forecasting or planning.',
                           'Full RAM capture and durable journaling add diagnostic overhead; primitive timings exclude them.',
                           'Installed PyTorch reports an Orin build-support warning; actual deterministic inference completed; no package changes.',
                           'No skipped inference, native CPU speed win, calibrated energy advantage or electrical spike trace.'])
    BASE.parent.mkdir(exist_ok=True);np.savez_compressed(BASE.with_suffix('.npz'),**data)
    meta['archive_sha256']=sha(BASE.with_suffix('.npz'));BASE.with_suffix('.json').write_text(json.dumps(meta,separators=(',',':'))+'\n')
    return audit_public()

def recount(meta,all_z):
    assert meta['status']=='CLOSED_STREAMING_PREDICTION_WITH_GATE_DISPOSITION'
    assert {r['name'] for r in meta['runs']}=={'smoke_s105101_learn','s105101_learn','s105101_frozen'}
    assert meta['admitted_seeds']==[105101] and meta['unrun_seeds']==[105102,105103]
    assert meta['skipped_inferences']==0 and not meta['pair']['skipping_enabled']
    for path,digest in meta['executed_source_hashes'].items():
        assert sha(ROOT/meta['executed_source_paths'].get(path,path))==digest,('executed source version differs',path)
    assert meta['protocol']==plan(),'prospective protocol changed'
    assert {r['name']:r['samples'] for r in meta['runs']}=={'smoke_s105101_learn':8,'s105101_learn':533,'s105101_frozen':533},'captured sample set incomplete'
    total=0;decoded={}
    for r in meta['runs']:
        z={k.split('__',1)[1]:v for k,v in all_z.items() if k.startswith(r['name']+'__')};n=r['samples']
        states=expand(z['initial'],{k:z['delta.'+k] for k in ['offsets','indices','values','sha256']},r['native_forwards'])
        assert r['native_forwards']==n*2==len(z['scores'])==len(z['inputs'])==len(z['labels'])
        assert z['clock_ns'].shape==(n,5) and np.all(np.diff(z['clock_ns'],axis=1)>0),'delayed-feedback causal clock violation'
        ref=Reference(z['initial'].tolist(),N,U,NW,5);cpu=RecentNearest();previous=z['initial'];prefix=None;prefix_scores=[]
        edits=0;post_edits=0
        for i,row in enumerate(r['rows']):
            o=observation(z['frames.previous'][i],z['frames.current'][i],[.15,row['observation']['action_y']],row['context'])
            assert o==row['observation'];x=encode(o);truth=band(z['frames.future'][i])
            assert truth==row['truth'] and row['nano_prediction']==band(z['frames.forecast'][i],False)
            assert row['persistence']==band(z['frames.current'][i])
            label=truth if row['teach'] and (r['arm']=='learn' or row['phase']=='prefix') else QUERY
            assert row['teacher']==label
            for j,teacher in [(2*i,QUERY),(2*i+1,label)]:
                assert np.array_equal(z['inputs'][j],x) and z['labels'][j]==teacher
                expected=ref.forward(x,teacher);assert expected==z['scores'][j].tolist(),'native output mismatch'
                assert ref.ram==states[j].tolist(),'native state mismatch'
                if teacher==QUERY:assert np.array_equal(previous,states[j]),'query updated RAM'
                pred=cpu.forward(x,teacher,row['context'])
                if j%2==0:assert pred==row['cpu_prediction'] and decode(expected)==(row['prediction'],row['score'],row['margin'])
                bit_edits=sum(int(v).bit_count() for v in (previous[2:].reshape(U,36)[:,4:]^states[j][2:].reshape(U,36)[:,4:]).ravel())
                edits+=bit_edits
                if row['phase']=='prefix':prefix=states[j].copy();prefix_scores.append(expected)
                else:
                    post_edits+=bit_edits
                    if r['arm']=='frozen':assert np.array_equal(states[j],prefix),'frozen weights changed'
                previous=states[j]
        assert edits==r['audit']['connection_bit_edits'] and post_edits==r['audit']['post_prefix_connection_bit_edits']
        assert r['model']['weights_unchanged'] and r['model']['model_calls']==n and r['device_released']
        assert r['model']['checkpoint_sha256']==meta['protocol']['checkpoint_sha256']
        for phase,m in r['audit']['metrics'].items():
            rr=[v for v in r['rows'] if v['phase']==phase];assert len(rr)==m['n']
            assert np.bincount([v['truth'] for v in rr],minlength=4).tolist()==m['histogram']
            for k in ['prediction','cpu_prediction','persistence','velocity_prediction','nano_prediction']:
                assert float(np.mean([v[k]==v['truth'] for v in rr]))==m[k]
        decoded[r['name']]=(z,states);total+=r['native_forwards']
    a,b=[decoded[n] for n in ['s105101_learn','s105101_frozen']]
    for key in ['frames.current','frames.previous','frames.future','frames.forecast','inputs']:
        assert np.array_equal(a[0][key],b[0][key]),'matched native arms have different streams'
    ar,br=[next(r for r in meta['runs'] if r['name']==n) for n in ['s105101_learn','s105101_frozen']]
    assert meta['pair']['learn']==ar['audit'] and meta['pair']['frozen']==br['audit'],'paired summary differs from reconstructed arms'
    prefix_n=sum(r['phase']=='prefix' for r in ar['rows'])*2
    assert np.array_equal(a[1][prefix_n-1],b[1][prefix_n-1]) and np.array_equal(a[0]['scores'][:prefix_n],b[0]['scores'][:prefix_n])
    assert meta['pair']['status']=='STOP_PREDICTION_GATE' and not all(meta['pair']['checks'].values())
    return dict(status='INDEPENDENT_PUBLIC_STREAM_PASS',physical_sessions=len(meta['runs']),native_forwards=total,
                complete_ram_snapshots=total,samples=sum(r['samples'] for r in meta['runs']),skipped_inferences=0,prediction_gate=meta['pair']['status'])

def audit_public():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==meta['archive_sha256']
    with np.load(BASE.with_suffix('.npz')) as z:return recount(meta,dict(z))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['export','audit-public']);a=p.parse_args()
    print(json.dumps(export() if a.operation=='export' else audit_public()))
