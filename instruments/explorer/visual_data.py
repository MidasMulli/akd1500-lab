"""Read-only RGB evidence profile. Never imports the SDK or starts hardware."""
from pathlib import Path
import argparse,base64,hashlib,io,json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'studies/20261026-visual-memory-v4'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(ram):
    a=ram[2:].reshape(220,36)
    return [dict(id=j,words=a[j,:4].astype('uint64').tolist(),connections=np.flatnonzero(np.unpackbits(a[j,4:].copy().view('u1'),bitorder='little')).tolist()) for j in range(220)]

def export(source=SOURCE):
    source=Path(source);assert json.loads((source/'MAC_REAUDIT.json').read_text())['status']=='MAC_SAVED_RAW_REAUDIT_PASS'
    result=dict(schema=3,profile='rgb-native-memory',title='AKD1500 / Visual memory explorer',live_compatible=False,
        evidence=dict(ram='Captured native full learner SRAM after every event',scores='Captured returned native overlap scores',features='Reconstructed fixed CNN features, four-image physical encoder qualification',gain='Derived difference from initial-mask response',prediction='Host class argmax from captured native scores',timing='Recorded event order, editorial playback timing',geometry='Logical record layout, not die coordinates'),
        limits=['One fixed seed/cohort','Initial old-view accuracy66.7% missed70%floor','Two separate chip runs, not simultaneous banks','Fixed encoder; supervised native memory learning','No electrical spike timing','No qualified live RGB actor in this Explorer profile'],
        source=dict(study='20261026-visual-memory-v4',mac_audit_sha256=sha(source/'MAC_REAUDIT.json'),assessment_sha256=sha(source/'FINAL_ASSESSMENT.json')),
        banks=[dict(id=0,name='Visual memory',units=220,plastic=list(range(200)),helpers=list(range(200,220)),classes=11,inputs=1024,num_weights=64,mapping=[3,3,0],thresholds=None,output_kind='raw_scores',class_labels=[f'Object {j+1}' for j in range(10)]+['Protected query records'])],
        mapping=json.loads((source/'native-fixtures/BUILD.json').read_text())['mapping'],cpu_controls=json.loads((source/'fresh-screen/RESULT.json').read_text())['controls'],images={},arms={})
    images={}
    for tag in ['adaptive','frozen']:
        f=source/'native-fixtures'/('fresh-'+tag);d=source/'raw'/('native-fresh-'+tag);r=json.loads((d/'AUDIT.json').read_text());assert r['status']=='EXACT_PHYSICAL_PASS'
        for name,h in r['hashes'].items():assert sha(d/name)==h,(tag,name)
        rows=json.loads((f/'ROWS.json').read_text());raw=np.fromfile(f/'cell.inputs','u1').reshape(520,128,128,3);x=np.fromfile(f/'cell.cpu_inputs','u1').reshape(520,1024);y=np.fromfile(f/'cell.labels','<i4');scores=np.fromfile(d/'result.scores','<i4').reshape(520,220);states=np.fromfile(d/'result.snapshots','<u4').reshape(520,-1);initial=np.fromfile(f/'cell.ram','<u4');before=initial;initial_records=records(initial);events=[]
        imasks=np.stack([np.array([int(k in set(rec['connections'])) for k in range(1024)],'i4') for rec in initial_records]);initial_scores=x.astype('i4')@imasks.T
        for i,row in enumerate(rows):
            old,new=records(before),records(states[i]);mask=np.stack([np.unpackbits(before[2:].reshape(220,36)[j,4:].copy().view('u1'),bitorder='little') for j in range(220)])
            assert np.array_equal(mask.astype('i4')@x[i].astype('i4'),scores[i]),('pre-edit native overlap',i)
            if y[i]==10:assert np.array_equal(before,states[i]),('read-only mutation',tag,i)
            changes=[]
            for j,(a,b) in enumerate(zip(old,new)):
                assert len(a['connections'])==len(b['connections'])==64
                if a!=b:changes.append(dict(bank=0,record=j,before=a,after=b,added=sorted(set(b['connections'])-set(a['connections'])),removed=sorted(set(a['connections'])-set(b['connections']))))
            key=f"s{row['session']}_f{row['frame']}_o{row['label']}";h=hashlib.sha256(raw[i].tobytes()).hexdigest()
            if key in images:assert images[key]==h
            else:
                images[key]=h;b=io.BytesIO();Image.fromarray(raw[i]).save(b,format='PNG');result['images'][key]=dict(data_url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),raw_rgb_sha256=h,width=128,height=128,session=row['session'],frame=row['frame'],object=row['label'])
            prediction=int(scores[i,:200].reshape(10,20).max(1).argmax());events.append(dict(index=i,phase=row['phase'],role='teach' if y[i]!=10 else 'query',teacher_enabled=bool(y[i]!=10),teaching_label=int(y[i]) if y[i]!=10 else None,evaluation_target=row['label'],prediction=prediction,image_id=key,changes=changes,counts=[dict(added=sum(len(c['added']) for c in changes),removed=sum(len(c['removed']) for c in changes))],derived_features=[np.flatnonzero(x[i]).tolist()],observed_potentials=[scores[i].tolist()],response_gain=[(scores[i]-initial_scores[i]).tolist()],policy_flags=None,memory_flags=None))
            before=states[i]
        result['arms'][tag]=dict(label='Native learning' if tag=='adaptive' else 'Trained then frozen',initial=[initial_records],events=events,assessment=r['metrics'],audit='EXACT_SAVED_NATIVE_SCORE_RAM_PASS',source_hashes={n:sha(d/n) for n in ['result.scores','result.snapshots','OWNER.json']})
    a,b=result['arms']['adaptive'],result['arms']['frozen']
    assert a['initial']==b['initial'] and all(a['events'][i]['image_id']==b['events'][i]['image_id'] for i in range(520))
    assert all(a['events'][i]['changes']==b['events'][i]['changes'] for i in range(80))
    result['audit']=dict(status='RGB_EXPLORER_EXPORT_PASS',events_per_arm=520,images=len(images),native_arms=2,matched_prefix_events=80,read_only_profile=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=export();a.out.parent.mkdir(exist_ok=True,parents=True);assert not a.out.exists();a.out.write_text(json.dumps(r,separators=(',',':')));receipt=dict(**r['audit'],data_sha256=sha(a.out),bytes=a.out.stat().st_size);a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':main()
