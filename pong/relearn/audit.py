"""Reconstruct every native update, match schedules and quantify history effects."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from pong.experiment import encode,save,sha
from pong.relearn.protocol import ARMS,N,U,PROTECTED,decode,schedule,stats,gate
from pong.relearn.model import RecentNearest


def summarize(plan,rows,scores,cpu,prefix_hash,prefix_scores_hash,edits):
    predictions=[decode(s) for s in scores];curves={};baseline={}
    for epoch in range(plan['adaptation_epochs']+1):
        phases=['new_A','retention_B']+(['fit_A'] if epoch else [])
        curves[str(epoch)]={p:stats(rows,predictions,p,epoch) for p in phases}
        baseline[str(epoch)]={p:stats(rows,cpu,p,epoch) for p in phases}
    return dict(old_A=stats(rows,predictions,'old_A',0),curves=curves,cpu_baseline=baseline,
        trained_ram_sha256=prefix_hash,prefix_scores_sha256=prefix_scores_hash,post_prefix_bit_edits=edits,
        final_predictions=predictions,native_forwards=len(rows))


def timing(events):
    # Prepared-input service primitives on Nano core2. This is not full system
    # latency or energy, and includes neither SSH nor the trace verification.
    replies=[e['response'] for e in events if e['label']==PROTECTED and e['index']>=512]
    def distribution(key):
        a=np.array([r[key] for r in replies],np.float64)
        return dict(n=len(a),median_ns=float(np.median(a)),p95_ns=float(np.percentile(a,95)),p99_ns=float(np.percentile(a,99)))
    result={key:distribution(key) for key in ['transaction_ns','native_thread_cpu_ns','cpu_reference_ns','cpu_task_ns']}
    result['scope']='Interleaved prepared-input primitives on same Nano core; native input MMIO/queue/output fetch; CPU baselines compute/optional learning. Excludes observation encoding, bit packing, SSH, complete RAM reads and JSON output. Not an end-to-end or energy advantage measurement.'
    return result


def audit(study,seed,nw,split='development'):
    plan=json.loads((study/'PROTOCOL.json').read_text());arms={};raw_audits=[];timings={}
    for arm in ARMS:
        name=f'{split}_nw{nw}_s{seed}_{arm}';raw=study/'raw'/name;capture=study/'runs'/name
        independent=verify(ROOT,raw);raw_audits.append(independent)
        rows=schedule(plan,arm,split);events=[json.loads(line) for line in (capture/'events.jsonl').read_text().splitlines()]
        scores=np.array(read_words(raw/'native.scores')).reshape(-1,U)
        xs=np.frombuffer((raw/'fixture.inputs').read_bytes(),np.uint8).reshape(-1,N);ys=read_words(raw/'fixture.labels')
        replies=[json.loads(line) for line in (raw/'native.stdout').read_text().splitlines() if json.loads(line)['kind']=='forward']
        assert len(rows)==len(events)==len(replies)
        baseline=RecentNearest();cpu=[];edit_count=0
        for i,(expected,event,reply) in enumerate(zip(rows,events,replies)):
            assert {k:event[k] for k in expected}==expected
            assert np.array_equal(encode(expected['observation']),xs[i]) and ys[i]==expected['label']
            for key in reply:assert event['response'][key]==reply[key]
            assert event['response']['scores']==scores[i].tolist()
            got=baseline.forward(xs[i],ys[i],expected['observation']['court']);cpu.append(got)
            assert got==event['cpu_prediction']==reply['cpu_task_prediction']
            if i>=512:edit_count+=reply['connection_bit_edits']
            if expected['label']==PROTECTED:
                before=raw/('fixture.ram' if i==0 else f'native.snapshot_{i-1}.ram')
                assert before.read_bytes()==(raw/f'native.snapshot_{i}.ram').read_bytes()
        result=summarize(plan,rows,scores,cpu,sha(raw/'native.snapshot_511.ram'),hashlib.sha256(scores[:512].astype('<i4').tobytes()).hexdigest(),edit_count)
        arms[arm]=result;timings[arm]=timing(events)
        save(capture/'AUDIT.json',dict(status='INDEPENDENT_THREE_ARM_AUDIT_PASS',raw=independent,timing=timings[arm],**result))
    disposition=gate(plan,arms)
    final=str(plan['adaptation_epochs']);fresh=arms['fresh']['curves'][final];learn=arms['relearn']['curves'][final]
    diagnosis=dict(status='POSTHOC_INTERPRETATION_OF_PROSPECTIVE_ARMS',
        fresh_fit=fresh['fit_A']['accuracy'],relearn_fit=learn['fit_A']['accuracy'],
        fresh_generalization=fresh['new_A']['accuracy'],relearn_generalization=learn['new_A']['accuracy'],
        history_gap=fresh['new_A']['accuracy']-learn['new_A']['accuracy'],
        fresh_fit_qualifies=fresh['fit_A']['accuracy']>=plan['gates']['fit_for_learnability_min'],
        boundary='Fresh differs in native learning history; a gap does not uniquely identify stale visible weights as the cause')
    report=dict(status='INDEPENDENT_THREE_ARM_COMPLETE',seed=seed,num_weights=nw,split=split,gate=disposition,diagnosis=diagnosis,arms=arms,timing=timings,raw_audits=raw_audits)
    save(study/'evidence'/f'TRIPLE_{split}_nw{nw}_s{seed}.json',report)
    print(json.dumps(dict(seed=seed,nw=nw,gate=disposition,diagnosis=diagnosis,
        final={a:arms[a]['curves'][final] for a in ARMS},cpu=arms['relearn']['cpu_baseline'][final])))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('seed',type=int);p.add_argument('--nw',type=int,default=64);p.add_argument('--split',default='development');a=p.parse_args();audit(a.study.resolve(),a.seed,a.nw,a.split)
