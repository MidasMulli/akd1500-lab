"""Scalar probabilities and before-query RAM decode, separate from Circuit."""
from pathlib import Path
import argparse,json,math
import numpy as np
from core import Layout,initial

B=Path(__file__).resolve().parents[1]

def recall(ram,k,g):
    stride=4+g.n1//32;weights=[sum(int(ram[2+j*stride+4+w])<<(32*w)for w in range(g.n1//32))for j in range(8)];counts=[]
    for c in range(3):
        bits=sum(1<<(4*int(i))for i in np.flatnonzero(k))|(1<<(4*(g.value+c)));counts.append(sum((bits&w).bit_count()>31 for w in weights))
    return counts.index(1)if counts.count(1)==1 and sum(counts)==1 else -1

def probability(logits,temp):
    survive=1.;goal=collision=0.
    for row in logits:
        a=[float(v)/temp for v in row];m=max(a);q=[math.exp(v-m)for v in a];den=sum(q);q=[v/den for v in q];goal+=survive*q[1];collision+=survive*q[2];survive*=q[0]
    return np.array([survive,goal,collision])

def run(tranche):
    data=B/('data-dev'if tranche=='DEV'else'data-held');receipt=json.loads((B/'data-dev/RECEIPT.json').read_text());count=0;maxerr=0.;streams=[]
    for p in sorted(B.glob(f'CPU_{tranche}_*/RESULT.json')):
        r=json.loads(p.read_text());g=Layout(r['candidate']);stem=f'STREAM_{tranche}_CHALLENGE_K{r["K"]}';rows=[json.loads(l)for l in(data/f'{stem}.jsonl').read_text().splitlines()];raw=np.load(data/f'{stem}.npz');z=np.load(p.parent/'TRACE.npz');indices=[i for i,v in enumerate(rows)if v['query']];assert np.array_equal(indices,z['episodes']);wm=[]
        for i in indices:
            row=rows[i];assert row['ep_index']==row['release_index']==i;s=np.array([row['wm_pN_T'],row['wm_pG_T'],row['wm_pC_T']]);q=probability(raw['wm_logits'][i],receipt['T_cal']);err=float(abs(s-q).max());maxerr=max(maxerr,err);assert err<2e-6;wm.append(s)
        wm=np.array(wm);ws,_,lt=g.parameters()
        for arm in ['all-normal','all-A-frozen','all-AB-frozen','all-retire-cut','error-normal','error-A-frozen']:
            before=initial(ws[1],lt[1])
            for j,k in enumerate(z['keys']):
                answer=recall(before,k,g);assert answer==z[f'found_{arm}'][j],(p.parent.name,arm,j);expected=wm[j]if answer<0 else .25*wm[j]+.75*np.eye(3)[answer];assert np.max(abs(expected-z[arm][j]))<1e-12;before=z[f'ram_{arm}'][j];count+=1
            for ph in range(6):
                take=z['phases']==ph;y=z['truth'][take];pred=z[arm][take];s=r['summary'][arm][f'phase{ph}'];assert len(y)==s['n']
                if len(y):assert abs(float((pred.argmax(1)==y).mean())-s['accuracy'])<1e-12 and abs(float(np.square(pred-np.eye(3)[y]).sum(1).mean())-s['brier'])<1e-12
        assert np.array_equal(z['truth'],[{'D':0,'S':1,'C':2}[rows[i]['outcome']]for i in indices]);streams.append(dict(profile=r['candidate'],K=r['K'],status=r['status'],queries=len(indices)))
    result=dict(status='PASS',tranche=tranche,software_RAM_query_replays=count,scalar_probability_max_error=maxerr,streams=streams,physical=False);(B/f'AUDIT_{tranche}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tranche',choices=['DEV','HELD']);a=p.parse_args();run(a.tranche)
