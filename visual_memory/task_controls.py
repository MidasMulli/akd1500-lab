"""Causal task controls on identical bits/teachers; no native-performance claim."""
import json
import numpy as np
from visual_memory.recruitment import S
from visual_memory.data import save,sha

def sequence(x,teachers,frozen=False):
    bank=[];labels=[];sums=np.zeros((10,512),np.int32);counts=np.zeros(10,np.int32);out=[];means=[];presented=0
    for v,y in zip(x,teachers):
        if bank:
            distances=np.logical_xor(np.array(bank),v).sum(1);out.append(labels[int(distances.argmin())])
            c=sums/np.maximum(counts[:,None],1);c=2*c-1;c/=np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-8);scores=c@(2*v.astype('i4')-1);scores[counts==0]=-np.inf;means.append(int(scores.argmax()))
        else:out.append(0);means.append(0)
        if y<10:
            if not frozen or presented<80:bank.append(v.copy());labels.append(int(y));sums[y]+=v;counts[y]+=1
            presented+=1
    assert len(bank)==(80 if frozen else 160)
    return np.array(out),np.array(means)

def review():
    result=dict(scope='Task correctness controls only; identical512binary descriptors and causal teacher transactions. No timing/energy comparison. Frozen controls stop after first80teachers.',cells={})
    for name in ['default','fresh-default']:
        f=S/'fixtures'/name;x=np.fromfile(f/'cell.inputs','u1').reshape(-1,512);teachers=np.fromfile(f/'cell.labels','<i4');rows=json.loads((f/'ROWS.json').read_text());cell={}
        for frozen in [False,True]:
            near,centroid=sequence(x,teachers,frozen)
            for method,pred in [('nearest_hamming',near),('cosine_class_mean',centroid)]:
                metrics={}
                for phase in ['old','before','after','retention','transfer']:
                    ii=np.array([i for i,r in enumerate(rows) if r['phase']==phase])
                    if len(ii):
                        correct=int(sum(pred[i]==rows[i]['label'] for i in ii));metrics[phase]=dict(correct=correct,total=len(ii),accuracy=correct/len(ii))
                cell[method+('_frozen' if frozen else '_adaptive')]=metrics
        result['cells'][name]=cell
    result['source_sha256']=sha(__file__);save(S/'TASK_CPU_CONTROLS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':review()
