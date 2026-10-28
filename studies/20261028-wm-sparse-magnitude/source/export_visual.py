"""Read-only native data -> deterministic 3D replay. No invented neuron taps."""
from pathlib import Path
import base64,hashlib,io,json,sys,zipfile
import numpy as np
from PIL import Image
from core import Layout
from encoder import Encoder
B=Path(__file__).resolve().parents[1];V=B/'visual';V.mkdir(exist_ok=True);g=Layout('oct9');enc=Encoder('oct9');cell=sys.argv[1]if len(sys.argv)>1 else'04';f=B/f'cell{cell}';archive=B/f'NATIVE_{cell}.zip';z=zipfile.ZipFile(archive);assert json.loads(z.read('RAW.QA.json'))['physical'];rows=json.loads((f/'ROWS.json').read_text());N=len(rows)
out=np.frombuffer(z.read('RAW.outputs'),'<u4').reshape(N,g.W);states=np.frombuffer(z.read('RAW.states1'),'<u4').reshape(N,-1);inputs=np.frombuffer(z.read('RAW.actual_inputs1'),'u1').reshape(-1,g.n1);episodes=[json.loads(l)for l in(B/'data-dev'/f'STREAM_DEV_CHALLENGE_K{4 if cell=="04"else 2}.jsonl').read_text().splitlines()];lat=np.load(B/'data-dev'/f'STREAM_DEV_CHALLENGE_K{4 if cell=="04"else 2}.npz');catalog=[]
for i,r in enumerate(episodes):
    key,_=enc.encode(lat['h0_pool'][i],lat['h5_pool'][i],int(r['program_id'][1:]));truth={'D':0,'S':1,'C':2}[r['outcome']];addresses=(4*np.flatnonzero(g.value_key(key,truth))).tolist()
    if not any(c['addresses']==addresses for c in catalog):catalog.append(dict(addresses=addresses,context=r['context_index'],regime=r['regime_AUDIT_ONLY'],outcome=truth,scene_id=r['scene_id']))
def memory(ram):
    a=ram[2:].reshape(8,26);result=[]
    for i,w in enumerate(a):
        addresses=np.flatnonzero(np.unpackbits(w[4:].copy().view('u1'),bitorder='little')).tolist();assert len(addresses)==32
        owner=next((j for j,c in enumerate(catalog)if c['addresses']==addresses),None);result.append(dict(id=i,addresses=addresses,learn_threshold=int(w[3]>>16),owner=owner))
    return result
before=np.frombuffer(z.read('RAW.initial1'),'<u4');at=0;events=[];flags=[0,0,0];first={}
for i,r in enumerate(rows):
    old=memory(before);new=memory(states[i]);before=states[i]
    if r['net']==0:flags=[int(out[i,g.P+g.error_probe]),int(out[i,g.P+g.correct_probe]),int(out[i,g.P+g.retire])]
    else:x=inputs[at];at+=1
    if r['episode']<0 or r['net']!=1:continue
    er=episodes[r['episode']]
    if er['context_index']!=0:continue
    first.setdefault(er['phase_AUDIT_ONLY'],r['episode'])
    if r['episode']!=first[er['phase_AUDIT_ONLY']]:continue
    changes=[]
    for j,(a,b)in enumerate(zip(old,new)):
        added=sorted(set(b['addresses'])-set(a['addresses']));removed=sorted(set(a['addresses'])-set(b['addresses']))
        if a!=b:changes.append(dict(record=j,added=added,removed=removed,threshold_before=a['learn_threshold'],threshold_after=b['learn_threshold']))
    response=out[i,:8].astype(int).tolist();assert set(response)<={0,1};active=np.flatnonzero(x).tolist();overlaps=[len(set(active)&set(q['addresses']))for q in old];assert response==[int(q>31)for q in overlaps]
    qidx=[j for j,q in enumerate(rows)if q['episode']==r['episode']and q['role']=='query'and q['net']==1];counts=[int(out[j,:8].sum())for j in qidx];qvalue=counts.index(1)if counts.count(1)==1 and sum(counts)==1 else -1
    e=dict(index=i,episode=r['episode'],phase=er['phase_AUDIT_ONLY'],regime=er['regime_AUDIT_ONLY'],role=r['role'],mode=r['mode'],candidate=r['candidate'],protocol_phase=r['phase'],truth={'D':0,'S':1,'C':2}[er['outcome']],wm=[er['wm_pN_T'],er['wm_pG_T'],er['wm_pC_T']],query_value=qvalue,flags=flags.copy(),active=active,before=old,memory=new,response=response,overlaps=overlaps,changes=changes)
    events.append(e)
assert at==len(inputs);images={};rgb=np.load(B/'data-dev/EXPLORER_FIXED_AB_DEV.npz')
for regime,gain in [('A','1.0'),('B','0.6')]:
    for name,key in [('input','obs_t5'),('imagined','imagined_t25'),('actual','true_final')]:
        raw=rgb[f'FIXED_AB_gain{gain}_{key}'][0];im=np.rint(255*np.clip(raw,0,1)).astype('u1')if raw.dtype.kind=='f'else raw;b=io.BytesIO();Image.fromarray(im).save(b,format='PNG');images[regime+'_'+name]=dict(src='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
assert all(episodes[e['episode']]['scene_id']==202610280100022 for e in events)
D=dict(schema=1,cell=cell,arm=json.loads((f/'FIXTURE.json').read_text())['arm'],catalog=catalog,events=events,images=images,geometry=dict(encoded_channels=176,content_neurons=8,source_mapping=[1,3,0],memory_mapping=[3,2,0],total_NPUs=6),source=dict(native_archive=archive.name,native_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),fixed_scene_id=202610280100022,first_admitted_prospectively=True),limits=['Recorded native replay; host-paced static role clocks and outcome feedback','Host frozen PCA/bin encoder; world-model weights unchanged','Memory stores sparse connections/outcome tags, not RGB images','Input highlights and overlaps reconstructed from captured inputs and RAM; eight return bits directly captured','Two source flags are measured population aggregates, not individual source-neuron taps','Four-context all-admit retention arm shown; separate error-only arm uses learned admission','Repeated cached challenge contexts, not independent natural video or fresh closed-loop control','Geometry and animation timing schematic; no energy, speed or GPU-concurrency claim'])
p=V/'data.json';p.write_text(json.dumps(D,separators=(',',':')));r=dict(status='NATIVE_WORLD_MEMORY_VISUAL_EXPORT_PASS',cell=cell,selected_events=len(events),source_forwards=N,all_displayed_memory_bits_native=True,all_8_outputs_check_against_actual_input_pre_RAM=True,actual_connection_changes=True,source_sha256=D['source']['native_sha256'],data_sha256=hashlib.sha256(p.read_bytes()).hexdigest());(V/'DATA_QA.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
