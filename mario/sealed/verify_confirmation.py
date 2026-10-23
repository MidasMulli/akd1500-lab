"""Inductive confirmation recount: train-only prefix, protected test, paired endpoints."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from bank import save, sha, initial

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words


def recount(study,train,test):
    tx=np.load(train/'inputs.npz',allow_pickle=False);ex=np.load(test/'inputs.npz',allow_pickle=False)
    train_rows=[r for r in json.loads((train/'EXPERIENCE.json').read_text())['rows'] if r['split']=='train' and r['valid']]
    test_rows=[r for r in json.loads((test/'EXPERIENCE.json').read_text())['rows'] if r['valid']]
    train_x=np.array([tx[f'{r["id"]}_{r["action"]}_projection'] for r in train_rows]);train_y=np.array([r['success'] for r in train_rows],np.int32)
    test_x=np.array([ex[f'{r["id"]}_{r["action"]}_projection'] for r in test_rows]);truth=np.array([r['success'] for r in test_rows]).reshape(-1,2)
    conditional=truth[:,0]!=truth[:,1]
    rows=[]
    for run in sorted((study/'raw').iterdir()):
        rec=json.loads((run/'RECEIPT.json').read_text());audit=verify(ROOT,run)
        if run.name=='smoke32':rows.append(dict(name=run.name,audit=audit));continue
        f=ROOT/rec['fixture'];meta=json.loads(f.with_suffix('.json').read_text());count=meta['training_presentations'];q=meta['evaluation_samples'];assert meta['prequeries']==0 and rec['samples']==count+q
        x=np.frombuffer(f.with_suffix('.inputs').read_bytes(),np.uint8).reshape(-1,1024);y=np.frombuffer(f.with_suffix('.labels').read_bytes(),'<i4');order=meta['order']
        expected_x=np.concatenate([train_x[order],test_x]);expected_y=np.concatenate([train_y[order],np.full(q,2)])
        if meta['arm']=='frozen':expected_y[:]=2
        assert np.array_equal(x,expected_x) and np.array_equal(y,expected_y),'training/test boundary or label leakage'
        original=read_words(f.with_suffix('.ram'));before=original
        for i in range(rec['samples']):
            state=read_words(run/f'native.snapshot_{i}.ram');assert state[2+128*36:]==original[2+128*36:]
            if y[i]==2:assert state==before,('protected query edited visible RAM',run.name,i)
            before=state
        if meta['arm']=='frozen':assert before==original
        score=np.array(read_words(run/'native.scores'),np.int32).reshape(-1,192)[-q:]
        actions=(score[:,64:128].max(1)-score[:,:64].max(1)).reshape(-1,2).argmax(1)
        correct=truth[np.arange(len(actions)),actions]
        rows.append(dict(name=run.name,seed=meta['seed'],arm=meta['arm'],actions=actions.tolist(),successes=int(correct.sum()),correct=correct.astype(int).tolist(),
            conditional_successes=int(correct[conditional].sum()),audit=audit,training_presentations=count,test_queries=q,test_queries_learn=False))
    cohort=[]
    for seed in range(82911,82916):
        arms={r['arm']:r for r in rows if r.get('seed')==seed}
        if set(arms)=={'full','frozen'}:
            cohort.append(dict(seed=seed,learned=arms['full']['successes'],frozen=arms['frozen']['successes'],delta=arms['full']['successes']-arms['frozen']['successes']))
    adequacy=dict(valid_contexts=len(truth),conditional_contexts=int(conditional.sum()),conditional_optimal_actions=truth[conditional].sum(0).astype(int).tolist())
    adequate=len(truth)>=80 and conditional.sum()>=24 and truth[conditional].sum(0).min()>=8
    result=dict(status='INDEPENDENT_CONFIRMATION_RECOUNT_PASS',runs=rows,cohorts=cohort,data_adequacy=adequacy,data_adequate=bool(adequate),
        best_constant=int(truth.sum(0).max()),constant_successes=truth.sum(0).astype(int).tolist(),
        train_test_encoded_overlap=sum(a.tobytes() in {b.tobytes() for b in train_x} for a in test_x),
        contexts=[{k:r[k] for k in ['id','level','wait','start']} for r in test_rows[::2]],test_truth=truth.astype(int).tolist(),
        source_sha256=sha(__file__))
    if len(cohort)==5:
        by_seed=[]
        for c in cohort:
            full=next(r for r in rows if r.get('seed')==c['seed'] and r.get('arm')=='full')
            frozen=next(r for r in rows if r.get('seed')==c['seed'] and r.get('arm')=='frozen')
            by_seed.append(np.array(full['correct'])-np.array(frozen['correct']))
        difference=np.mean(by_seed,axis=0)
        groups={}
        for i,r in enumerate(test_rows[::2]):groups.setdefault((r['level'],r['wait']),[]).append(i)
        keys=list(groups);totals=np.array([[difference[v][conditional[v]].sum(),conditional[v].sum()] for v in groups.values()])
        rng=np.random.default_rng(83001);draw=[];hier=[]
        while len(draw)<10000:
            a=totals[rng.integers(len(keys),size=len(keys))].sum(0)
            if a[1]:draw.append(a[0]/a[1])
        levels=sorted({k[0] for k in keys})
        while len(hier)<10000:
            chosen=[]
            for level in rng.choice(levels,size=len(levels),replace=True):
                options=[j for j,k in enumerate(keys) if k[0]==level];chosen.extend(rng.choice(options,size=len(options),replace=True))
            a=totals[chosen].sum(0)
            if a[1]:hier.append(a[0]/a[1])
        interval=np.quantile(draw,[.025,.975]).tolist();conservative=np.quantile(hier,[.025,.975]).tolist()
        gain=float(difference.mean());wins=sum(c['delta']>0 for c in cohort)
        result.update(learning_effect_gate='PASS' if adequate and wins>=4 and gain>=.05 and interval[0]>0 else 'FAIL',
            initialization_wins=wins,all_valid_gain=gain,conditional_gain=float(difference[conditional].mean()),
            block_bootstrap_95=interval,hierarchical_bootstrap_95=conservative,
            learned_total=sum(c['learned'] for c in cohort),frozen_total=sum(c['frozen'] for c in cohort),opportunities=5*len(truth),
            best_constant_total=5*int(truth.sum(0).max()),usefulness='NOT_QUALIFIED_WITHOUT_FULL_PHYSICAL_BASELINES')
    save(study/'evidence/RECOUNT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','contexts','test_truth']}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('train',type=Path);p.add_argument('test',type=Path);a=p.parse_args();recount(a.study.resolve(),a.train.resolve(),a.test.resolve())
