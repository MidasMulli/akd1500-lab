"""Prospective causal CPU screen. Outcomes are released only after prediction.

The native-rule arms return answers only through their emulated learning RAM.
Task-level CPU stores are explicitly separate controls, not chip claims.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from core import Circuit,Layout
from encoder import Encoder

B=Path(__file__).resolve().parents[1]
LABEL={'D':0,'S':1,'C':2}
ARMS=['all-normal','all-A-frozen','all-AB-frozen','all-retire-cut','error-normal','error-A-frozen']

def pack(k):return sum(1<<int(i)for i in np.flatnonzero(k))

class Store:
    def __init__(self,kind):self.kind=kind;self.items=[]
    def query(self,k,raw,program):
        p=pack(k);eligible=[v for v in self.items if v[2]==program]
        if self.kind=='float':return min(reversed(eligible),key=lambda v:float(np.square(v[3]-raw).sum()))[1]if eligible else None
        if self.kind=='hamming':return min(reversed(eligible),key=lambda v:(p^v[0]).bit_count())[1]if eligible else None
        for i,v in enumerate(self.items):
            if v[0]==p:
                if self.kind=='lru':self.items.append(self.items.pop(i))
                return v[1]
        return None
    def feedback(self,k,raw,program,truth):
        p=pack(k)
        same=lambda v:v[2]==program and (np.array_equal(v[3],raw)if self.kind=='float'else v[0]==p)
        existing=[i for i,v in enumerate(self.items)if same(v)]
        if existing:
            i=existing[0];old=self.items[i];self.items[i]=(p,truth,program,raw.copy())
            if self.kind in ['lru','hamming','float']:self.items.append(self.items.pop(i))
        elif len(self.items)<8 or self.kind!='exact':
            self.items.append((p,truth,program,raw.copy()));self.items=self.items[-8:]

def metrics(p,y,take):
    p=np.asarray(p)[take];y=np.asarray(y)[take]
    if not len(y):return dict(n=0,accuracy=None,brier=None)
    return dict(n=len(y),accuracy=float((p.argmax(1)==y).mean()),brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()))

def run(data,out,candidate,tranche,condition,capacity):
    # V3 names the selected population CHALLENGE; retain original filenames.
    assert condition=='INFORMATIVE'
    stem=f'STREAM_{tranche}_CHALLENGE_K{capacity}';rows=[json.loads(l)for l in(data/f'{stem}.jsonl').read_text().splitlines()];z=np.load(data/f'{stem}.npz')
    enc=Encoder(candidate);out.mkdir(parents=True,exist_ok=False);circuits={a:Circuit(Layout(candidate))for a in ARMS}
    for c in circuits.values():c.bootstrap()
    stores={a:Store(kind)for a,kind in [('exact8','exact'),('recentHamming8','hamming'),('recentFloat8','float'),('FIFO8','fifo'),('LRU8','lru')]}
    pred={a:[]for a in ['calibratedWM','uncalibratedWM',*ARMS,*stores]};found={a:[]for a in [*ARMS,*stores]};states={a:[]for a in ARMS};truths=[];phases=[];episodes=[];keys=[];projections=[]
    for i,r in enumerate(rows):
        assert r['ep_index']==r['release_index']==i
        if not r['query']:continue
        h0,h5=z['h0_pool'][i],z['h5_pool'][i];assert np.isfinite(h0).all()and np.isfinite(h5).all()
        program=int(r['program_id'][1:]);k,v=enc.encode(h0,h5,program);raw=np.r_[h0,h5,h5-h0].astype('f8')
        wm=np.array([r['wm_pN_T'],r['wm_pG_T'],r['wm_pC_T']]);uncal=np.array([r['wm_pN'],r['wm_pG'],r['wm_pC']]);assert abs(wm.sum()-1)<1e-6
        # Read every answer before releasing the actual outcome of this episode.
        answer={a:c.query(k)for a,c in circuits.items()};answer.update({a:s.query(k,raw,program)for a,s in stores.items()})
        pred['calibratedWM'].append(wm.copy());pred['uncalibratedWM'].append(uncal.copy())
        for a,value in answer.items():
            found[a].append(-1 if value is None else value);pred[a].append(wm.copy()if value is None else .25*wm+.75*np.eye(3)[value])
        # Outcome and audit phase enter only feedback or prespecified controls.
        truth=LABEL[r['outcome']];phase=int(r['phase_AUDIT_ONLY']);wc=int(wm.argmax())
        for a,c in circuits.items():
            freeze=(a.endswith('A-frozen')and phase>=1)or(a.endswith('AB-frozen')and phase>=2)
            c.feedback(k,wc,truth,cut=a.endswith('retire-cut'),freeze=freeze,admit_all=a.startswith('all-'));states[a].append(c.memory.ram.copy())
        for s in stores.values():s.feedback(k,raw,program,truth)
        truths.append(truth);phases.append(phase);episodes.append(i);keys.append(k);projections.append(v)
    y=np.asarray(truths);ph=np.asarray(phases);mask={'all':np.ones(len(y),bool),**{f'phase{j}':ph==j for j in range(6)},'returnedA':np.isin(ph,[2,4]),'returnedB':np.isin(ph,[3,5])}
    summary={a:{n:metrics(p,y,t)for n,t in mask.items()}for a,p in pred.items()}
    pure={a:{n:dict(n=int(t.sum()),correct=int((np.asarray(vals)[t]==y[t]).sum()),hits=int((np.asarray(vals)[t]>=0).sum()),accuracy=float((np.asarray(vals)[t]==y[t]).mean())if t.any()else None)for n,t in mask.items()}for a,vals in found.items()}
    base=summary['calibratedWM'];checks={}
    for j in [2,3,4,5]:checks[f'pure_all_phase{j}']=pure['all-normal'][f'phase{j}']['accuracy']>=.95
    for a,fr in [('all-normal','all-A-frozen'),('error-normal','error-A-frozen')]:
        s=summary[a];checks[f'{a}_returnedA']=s['returnedA']['brier']<=base['returnedA']['brier']+.01
        checks[f'{a}_returnedB_Brier']=s['returnedB']['brier']<=base['returnedB']['brier']-.10
        checks[f'{a}_returnedB_accuracy']=s['returnedB']['accuracy']>=base['returnedB']['accuracy']+.20
        checks[f'{a}_vs_A_frozen']=s['returnedB']['accuracy']>=summary[fr]['returnedB']['accuracy']+.10
    checks['AB_frozen_noninferiority']=summary['all-normal']['returnedB']['accuracy']>=summary['all-AB-frozen']['returnedB']['accuracy']
    sets={}
    for k,t in zip(keys,y):sets.setdefault(pack(k),set()).add(int(t))
    result=dict(status='PASS'if all(checks.values())else'FAILED_GATE',candidate=candidate,tranche=tranche,condition=condition,K=capacity,checks=checks,summary=summary,pure_memory=pure,total_episodes=len(rows),queried_episodes=len(y),probe_terminal_episodes=len(rows)-len(y),unique_keys=len(sets),conflicting_keys=sum(len(v)>1 for v in sets.values()),physical_sessions=0,query_before_release=True,capacity=8)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'TRACE.npz',truth=y,phases=ph,episodes=np.array(episodes),keys=np.array(keys),projections=np.array(projections),**{a:np.array(p)for a,p in pred.items()},**{f'found_{a}':np.array(vals)for a,vals in found.items()},**{f'ram_{a}':np.array(v)for a,v in states.items()})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate',choices=['quad13','oct9']);p.add_argument('condition',choices=['INFORMATIVE','UNINFORMATIVE']);p.add_argument('K',type=int,choices=[2,4,8]);p.add_argument('--tranche',choices=['DEV','HELD'],default='DEV');a=p.parse_args();t=time.monotonic();r=run(B/('data-dev'if a.tranche=='DEV'else'data-held'),B/f'CPU_{a.tranche}_{a.candidate}_{a.condition}_K{a.K}',a.candidate,a.tranche,a.condition,a.K);print(json.dumps({k:r[k]for k in ['status','candidate','condition','K','checks','unique_keys','conflicting_keys','queried_episodes']},indent=2));print('wall_s',time.monotonic()-t)
