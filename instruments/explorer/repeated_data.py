"""Read-only binding of native repeated-image memory into the existing Explorer."""
from pathlib import Path
import argparse,base64,hashlib,io,json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'studies/20261027-repeated-native-memory'
CELLS={'native':('development02','Native learned manager'),'policy_frozen':('policy_frozen01','Policy frozen'),'retire_cut':('retire_cut01','Retirement inhibited'),'content_frozen':('content_frozen01','Content frozen'),'normal_cap':('normal_cap01','Without capacity reset'),'prior_cap24':('prior_cap2401','Prior threshold setting')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(ram,n,u):
    a=ram[2:].reshape(u,4+n//32)
    return [dict(id=j,words=a[j,:4].astype('uint64').tolist(),connections=np.flatnonzero(np.unpackbits(a[j,4:].copy().view('u1'),bitorder='little')).tolist())for j in range(u)]

def export(source=SOURCE):
    source=Path(source);cohort=source/'images-development';manifest=json.loads((cohort/'IMAGES.json').read_text());rgb=np.load(cohort/'RAW_IMAGES.npy');keys=np.load(cohort/'DESCRIPTORS.npy')
    images={}
    for j,im in enumerate(rgb):
        assert hashlib.sha256(im.tobytes()).hexdigest()==manifest['raw_rgb_sha256'][j]
        b=io.BytesIO();Image.fromarray(im).save(b,format='PNG');images['episode'+str(j)]=dict(data_url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),raw_rgb_sha256=manifest['raw_rgb_sha256'][j],object=manifest['objects'][j],session=manifest['session'],frame=manifest['frame'],width=128,height=128)
    result=dict(schema=4,profile='repeated-native-memory',live_compatible=False,title='Native repeated visual memory',images=images,keys=keys.tolist(),
        banks=[dict(id=0,name='Learned control',units=96,plastic=[0,48],helpers=[j for j in range(96)if j not in [0,48]],classes=2,inputs=1024,num_weights=32,mapping=[1,3,0],thresholds=[31 if j in [0,48]else 0 for j in range(96)]),dict(id=1,name='Content memory',units=8,plastic=list(range(8)),helpers=[],classes=1,inputs=256,num_weights=32,mapping=[3,1,0],thresholds=[31]*8)],
        mapping=[dict(layer=label,nps=[dict(np=npid)])for label,npid in [('Policy learner',[1,3,0]),('Predicates',[2,1,0]),('Content isolation',[2,2,0]),('Native mailbox',[2,3,0]),('Content pool',[3,1,0]),('Recall return',[3,2,0])]],
        source=dict(study=source.name,descriptor_sha256=sha(cohort/'DESCRIPTORS.npy'),encoder_sha256=manifest['fixed_encoder_sha256']),
        limits=['Recorded realcamera images; frozen host CNN and31projections','Exact image-episode descriptors, not unseenviewrecognition','Host supplies policyteachers and fixedrole/configuration clocks','Card-side scratch mailbox, distinct from NPU learnerRAM','Returned binary responses and captured RAM; overlaps reconstructed, geometry schematic','AdaptiveCPUtaskcontrols also pass; no efficiency or priority claim'],
        arms={})
    for tag,(name,label)in CELLS.items():
        f=source/name
        if not (f/'native.AUDIT.json').exists():continue
        a=json.loads((f/'native.AUDIT.json').read_text());assert a['status']=='RAW_NATIVE_AUDIT_PASS'
        rows=json.loads((f/'ROWS.json').read_text());n=len(rows);outputs=np.fromfile(f/'native.outputs','<u4').reshape(n,256)
        states=[np.fromfile(f/f'native.states{k}','<u4').reshape(n,3458 if k==0 else 98)for k in range(2)]
        before=[np.fromfile(f/f'bank{k}.ram','<u4')for k in range(2)];initial=[records(before[0],1024,96),records(before[1],256,8)]
        actual=[np.fromfile(f/f'native.actual_inputs{k}','u1').reshape(-1,1024 if k==0 else 256)for k in range(2)]
        pots=[np.fromfile(f/f'native.derived_potentials{k}','<i4').reshape(-1,96 if k==0 else 8)for k in range(2)]
        at=[0,0];events=[];last_control=[0]*4
        for i,row in enumerate(rows):
            k=row['net'];x=actual[k][at[k]].copy()
            if k==0:x[960:992]=0
            assert x.max()<=1
            old=[records(before[0],1024,96),records(before[1],256,8)];new=[records(states[0][i],1024,96),records(states[1][i],256,8)]
            scores=np.array([sum(map(int,x[rec['connections']]))for rec in old[k]],'i4');np.testing.assert_array_equal(scores,pots[k][at[k]]);at[k]+=1
            response=[None]*96 if k==0 else [None]*8;fresh=[None,None]
            if k==0:
                response[0]=int(outputs[i,111]);response[48]=int(outputs[i,112]);assert response[0]==int(scores[0]>31)and response[48]==int(scores[48]>31);last_control=outputs[i,110:114].tolist()
            else:
                response=outputs[i,:8].tolist();np.testing.assert_array_equal((scores>31).astype('i4'),response)
            fresh[k]=response;features=[[],[]];features[k]=np.flatnonzero(x).tolist();derived=[[0]*96,[0]*8];derived[k]=scores.tolist();changes=[]
            for bank in range(2):
                for j,(u,v)in enumerate(zip(old[bank],new[bank])):
                    assert len(u['connections'])==len(v['connections'])==32
                    if u!=v:changes.append(dict(bank=bank,record=j,before=u,after=v,added=sorted(set(v['connections'])-set(u['connections'])),removed=sorted(set(u['connections'])-set(v['connections']))))
            winner=int(np.argmax(response))if k==1 and sum(response)==1 else None;recalled=None
            if winner is not None:
                mask=set(old[1][winner]['connections']);matches=[j for j,key in enumerate(keys)if mask==set((4*np.flatnonzero(key)).tolist())]
                recalled=matches[0]if len(matches)==1 else None
            events.append(dict(index=i,phase=row['role'],role=row['role'],cycle=row['cycle'],key=row['key'],quality=row['quality'],teacher_enabled=(k==0 and row['mode']==0)or(k==1 and row['mode']!=0),teaching_label=row['label'],image_id=None if row['key']is None else 'episode'+str(row['key']),active_bank=k,changes=changes,counts=[dict(added=sum(len(c['added'])for c in changes if c['bank']==b),removed=sum(len(c['removed'])for c in changes if c['bank']==b))for b in range(2)],derived_features=features,derived_potentials=derived,native_responses=fresh,control_flags=last_control.copy(),recalled_key=recalled,returned_record=winner))
            before=[states[0][i],states[1][i]]
        hashes={name:sha(f/name)for name in ['native.outputs','native.states0','native.states1','native.actual_inputs0','native.actual_inputs1','native.descriptors','native.AUDIT.json']}
        result['arms'][tag]=dict(label=label,initial=initial,events=events,assessment=a,audit='EXACT_NATIVE_INPUT_RETURN_RAM_PASS',source_hashes=hashes)
    assert 'native'in result['arms']
    result['cpu_controls']=json.loads((source/'development02/TASK_CPU.json').read_text())
    result['audit']=dict(status='REPEATED_EXPLORER_NATIVE_EXPORT_PASS',native_arms=len(result['arms']),images=len(images),primary_events=len(result['arms']['native']['events']),all_displayed_responses_native=True,overlaps_reconstructed=True,geometry_schematic=True,live_compatible=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=export(a.source);assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,separators=(',',':')));receipt=dict(**r['audit'],data_sha256=sha(a.out),bytes=a.out.stat().st_size);a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
