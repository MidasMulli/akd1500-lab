"""Export observed words/RAM edits; no reference predictions as film outputs."""
from pathlib import Path
import argparse,json
import numpy as np
from visual_memory import silhouette as m
from visual_memory.data import save,sha
def bits(words):return [k*32+b for k,v in enumerate(words) for b in range(32) if int(v)>>b&1]
def extract(f,full=True):
    a=json.loads((f/'native.AUDIT.json').read_text());assert a['status']=='MEASURED_NATIVE_PASS'
    rr=json.loads((f/'ROWS.json').read_text());n=len(rr);yy=np.fromfile(f/'native.native_payloads','<u4').reshape(n,512);xx=np.fromfile(f/'native.actual_inputs','u1').reshape(n,2048);ss=[np.memmap(f/f'native.states{k}',dtype='<u4',mode='r',shape=(n,len(np.fromfile(f/f'bank{k}.ram','<u4')))) for k in range(2)];before=[np.fromfile(f/f'bank{k}.ram','<u4') for k in range(2)]
    initial=[bits(before[1][2:].reshape(400,20)[j,4:]) for j in range(384)];policy_initial=[bits(before[0][2:].reshape(32,68)[j,4:]) for j in [8,9,10,11]];frames=[];activations=np.fromfile(f/'native.derived_activations0','u1').reshape(n,32) if full else None
    for i,(r,y,x) in enumerate(zip(rr,yy,xx)):
        z=x.reshape(512,4)[:,0];e=dict(index=i,phase=r['phase'],role=r['role'],pattern=r.get('pattern'),source=r.get('source'),input=z[:288:3].tolist(),known=(z[:288:3]|z[288:384]).tolist(),output=y[:288:3].tolist(),ready=int(y[472]),ids=y[468:472].tolist(),responses=y[384:448].tolist(),query_requested=int(z[m.Q]),candidate=r.get('candidate'),bit=r.get('bit'),policy=y[448:452].tolist() if r['role']=='policy_probe' else None,changes=[],stored_updates=[],policy_changes=[],added=0,cue_added=0,kind=r['source_kind'])
        if full:
            coded=m.fixed(m.fixed(activations[i],m.matrices()[0]),m.matrices()[1]);e['derived_features']=np.flatnonzero(coded).tolist();e['feedback']=r.get('good') if r['role']=='policy_teacher' else None
            for k in range(2):
                stride=68 if k==0 else 20;old=before[k][2:].reshape(-1,stride)[:,4:];new=ss[k][i,2:].reshape(-1,stride)[:,4:]
                for j in np.flatnonzero(np.any(old!=new,axis=1)):
                    aa=set(bits(old[j]));bb=set(bits(new[j]));change=dict(cell=int(j),added=sorted(bb-aa),removed=sorted(aa-bb))
                    if k==1 and j<384:
                        e['changes'].append(change);e['added']+=len(change['added']);obj=int(j)//96;b=int(j)%96;key=set(range(obj*32,(obj+1)*32-1));v=0 if bb==key|{128} else 1 if bb==key|{129} else -1;e['stored_updates'].append([obj,b,v])
                    if k==0 and j in m.POLICY_CELLS:e['policy_changes'].append(change);e['cue_added']+=len(change['added'])
                before[k]=ss[k][i]
            if r['role']=='content_teacher':assert int(z[487])==r['candidate'][r['bit']] and z[486]+z[487]==1
        frames.append(e)
    return dict(initial=initial,policy_initial=policy_initial,frames=frames,audit=a)
def main(root):
    out=root/'visual';out.mkdir(exist_ok=True);assert not (out/'data.json').exists();sessions={name:extract(root/name,name=='primary') for name in ['primary','open','manager_frozen']};ff=sessions['primary']['frames'];schedule=[]
    def put(indices,duration,section,emphasize=False,compare=False):
        weights=[45 if emphasize and ff[i]['added'] else 1 for i in indices];total=sum(weights)
        for i,w in zip(indices,weights):schedule.append(dict(source='primary',index=i,duration=duration*w/total,section=section,compare=compare))
    phase=lambda s:[i for i,e in enumerate(ff) if e['phase']==s]
    fullq=lambda s:[i for i in phase(s) if ff[i]['role']=='query' and all(ff[i]['known'])]
    put([i for i in phase('bootstrap') if i>=18],8,'Teach')
    put(fullq('old_probe'),2,'Recall')
    put(phase('misleading_old'),4,'Reject misleading updates')
    put(phase('new_images'),6,'Add new memories')
    put(fullq('four_resident'),2,'Recall all four')
    put(phase('quality_reversal'),3,'Update the source check')
    put(phase('legitimate_revision'),6,'Accept a genuine revision',True)
    put(fullq('after_revision'),2,'Recall the revised memories')
    partial=[]
    for obj in range(4):partial.append(next(i for i in phase('final') if ff[i]['role']=='query' and ff[i]['pattern']==obj and not all(ff[i]['known']) and ff[i]['ready']))
    put(partial,4.8,'Recall from partial images')
    abstain=next(i for i in phase('final') if ff[i]['role']=='query' and not ff[i]['ready']);put([abstain],1.4,'Unresolved cue')
    unknown=next(i for i in phase('final') if ff[i]['role']=='unknown');put([unknown],.8,'Unfamiliar input')
    put([fullq('final')[0]],3,'Same query, different memory control',compare=True)
    put(phase('feedback'),2.4,'Native output returns as input')
    d=dict(version=1,names=m.NAMES,observed_cells=m.OBSERVED_CELLS.tolist(),sessions=sessions,schedule=schedule,provenance=dict(outputs='Actual native int32 words read before SDKfetch',learning='Full post-forward native RAM snapshots',responses='64 direct binary cell probes per forward; source probes only on explicit read-only probe rows',stored_panel='Derived exact image-bit template decode from native RAM, separately labelled from returned pixels',timing='Editorial replay, no electrical or device-latency timing claim',allocation='Record/pixel slots externally assigned; learned admission, not learned allocation'))
    save(out/'data.json',d);save(out/'DATA_MANIFEST.json',dict(status='OBSERVED_NATIVE_DATA_EXPORT',source_hashes={str(root/name/p):sha(root/name/p) for name in sessions for p in ['native.native_payloads','native.states0','native.states1','native.actual_inputs','ROWS.json','native.AUDIT.json']},data_sha256=sha(out/'data.json'),duration_s=sum(r['duration'] for r in schedule),display_schedule_events=len(schedule),total_native_primary_events=len(ff),methods=d['provenance']));print(json.dumps(dict(out=str(out),bytes=(out/'data.json').stat().st_size,duration=sum(r['duration'] for r in schedule))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
