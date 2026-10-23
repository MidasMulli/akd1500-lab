"""Three histories, identical new-rule experiences, no post-prefix B teaching."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pong.experiment import encode,outcome,make_cases,save

N,U,CLASSES,PROTECTED=1024,160,5,4
ARMS=['frozen','fresh','relearn']


def design():
    return dict(version=1,development_seeds=[85301,85302,85303],confirmation_seeds=[85401,85402,85403],
        baseline_num_weights=64,controlled_candidates=[32,48],
        fixed_parameters=dict(initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25),
        initial=make_cases(94001,64),adaptation=[r for r in make_cases(94002,64) if r['court']==0],
        development=make_cases(94003,64),confirmation=make_cases(94004,96),
        prefix_epochs=4,adaptation_epochs=6,
        arms=dict(frozen='B original training, A original training, then protected queries only',
            fresh='B original training, matched protected A prefix, then new-rule A training',
            relearn='B original training, A original training, then identical new-rule A training'),
        frozen_control='Frozen and relearn share every original prefix input and label; full trained RAM and outputs must match',
        fresh_control='No exposure to old A labels; exactly the same new A teaching cases, repetitions and query order as relearn. This changes learning history, not only visible stale weights',
        retention='All B labels after prefix_B are protected in all arms, including during A adaptation',
        primary='Final-epoch changed-A development accuracy, measured on separate trajectories with no teaching',
        gates=dict(new_accuracy_min=.65,new_gain_over_frozen_min=.10,new_mae_ratio_max=.80,
            old_accuracy_min=.65,retention_initial_min=.65,retention_loss_max=.10,
            fit_for_learnability_min=.90,seed_wins_required=3),
        cpu_baseline='Per-court 64-example FIFO nearest neighbour on the exact binary inputs; newest breaks distance ties; same allowed labels; no future physics access',
        staged_order=['reference replay validation','native smoke','one baseline three-arm development seed',
            'review baseline diagnosis','if failed, exact-reference CPU screen of only connection counts32 and48',
            'admit at most one qualifying candidate on its first development seed',
            'only after first seed passes, remaining two development seeds',
            'only after all development seeds pass, fresh confirmation seeds one at a time',
            'only after all confirmation seeds pass, wall-paced paddle-control comparison'],
        selection='One-factor connection count only. At most one candidate may be selected; choose highest worst-seed changed accuracy among candidates passing every development gate; tie fewer connections. No passing candidate means stop hardware expansion.',
        claims=['Same chip across seeds; no independent-device claim',
            'Inputs are host observed-motion codes; no pixel learning or native reinforcement-learning claim',
            'CPU algorithm screens are predictions, not physical measurements',
            'Query-only means unchanged visible RAM; hidden selector history is replayed, not restored',
            'No energy or speed advantage inferred from predictive accuracy'])


def schedule(plan,arm,split='development'):
    assert arm in ARMS and split in ['development','confirmation']
    rows=[]
    def add(phase,cases,changed=False,teach=False,epoch=0):
        for c in cases:
            truth,landing=outcome(c,changed)
            rows.append(dict(index=len(rows),phase=phase,epoch=epoch,observation=c,changed=changed,
                truth=truth,landing=landing,label=truth if teach else PROTECTED))
    for court in [1,0]:
        cases=[c for c in plan['initial'] if c['court']==court]
        for epoch in range(plan['prefix_epochs']):
            order=np.random.default_rng(95000+court*100+epoch).permutation(len(cases))
            add('prefix_B' if court==1 else 'prefix_A',[cases[int(i)] for i in order],teach=court==1 or arm!='fresh')
    a=[c for c in plan[split] if c['court']==0];b=[c for c in plan[split] if c['court']==1]
    add('old_A',a);add('retention_B',b);add('new_A',a,True)
    for epoch in range(1,plan['adaptation_epochs']+1):
        order=np.random.default_rng(96000+epoch).permutation(len(plan['adaptation']))
        add('teach_A',[plan['adaptation'][int(i)] for i in order],True,arm!='frozen',epoch)
        add('fit_A',plan['adaptation'],True,epoch=epoch)
        add('new_A',a,True,epoch=epoch);add('retention_B',b,epoch=epoch)
    assert len(rows)<=4096
    assert all(r['label']==PROTECTED for r in rows if r['observation']['court']==1 and r['phase']!='prefix_B')
    return rows


def stats(rows,predictions,phase,epoch):
    selected=[(r,p) for r,p in zip(rows,predictions) if r['phase']==phase and r['epoch']==epoch]
    assert selected
    return dict(n=len(selected),correct=sum(r['truth']==p for r,p in selected),
        accuracy=float(np.mean([r['truth']==p for r,p in selected])),
        landing_mae=float(np.mean([abs((p+.5)/4-r['landing']) for r,p in selected])))


def decode(scores):return int(np.asarray(scores).reshape(CLASSES,U//CLASSES)[:4].max(1).argmax())


def gate(plan,arms):
    a,b=arms['relearn'],arms['frozen'];g=plan['gates'];final=plan['adaptation_epochs']
    x,y=a['curves'][str(final)]['new_A'],b['curves'][str(final)]['new_A']
    checks=dict(shared_prefix_ram=a['trained_ram_sha256']==b['trained_ram_sha256'],
        shared_prefix_scores=a['prefix_scores_sha256']==b['prefix_scores_sha256'],
        frozen_ram_unchanged=b['post_prefix_bit_edits']==0,
        original_A=a['old_A']['accuracy']>=g['old_accuracy_min'],
        original_B=a['curves']['0']['retention_B']['accuracy']>=g['retention_initial_min'],
        changed_A=x['accuracy']>=g['new_accuracy_min'],
        changed_gain=x['accuracy']-y['accuracy']>=g['new_gain_over_frozen_min'],
        changed_mae=x['landing_mae']<=g['new_mae_ratio_max']*y['landing_mae'],
        retention=a['curves'][str(final)]['retention_B']['accuracy']>=a['curves']['0']['retention_B']['accuracy']-g['retention_loss_max'])
    return dict(status='PASS' if all(checks.values()) else 'STOP_DEVELOPMENT_GATE',checks=checks,
        changed_gain=x['accuracy']-y['accuracy'],
        fresh_fit_accuracy=arms['fresh']['curves'][str(final)]['fit_A']['accuracy'],
        fresh_new_accuracy=arms['fresh']['curves'][str(final)]['new_A']['accuracy'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);save(p.parse_args().output,design())
