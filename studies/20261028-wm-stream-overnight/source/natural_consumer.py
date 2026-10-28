"""N-only CPU qualification; explicit skips, unchanged native-rule learner."""
import argparse,json,time
from pathlib import Path
import numpy as np
from live_consumer import (Circuit,Layout,Encoder,Store,LABEL,check_deadline,append,atomic,validate_query,next_line,fuse)

ARMS=['error-only','all-admit','after-A-frozen']
def read_rows(p):return p.read_text().splitlines() if p.exists() else []
def next_event(d,seq,qr,sr):
    start=time.monotonic()
    while True:
        check_deadline();q=read_rows(d/'QUERY.jsonl');s=read_rows(d/'SKIP.jsonl')
        assert len(q)<=qr+1 and len(s)<=sr+1, 'duplicate producer line'
        qp=len(q)>qr;sp=len(s)>sr;assert not(qp and sp),'both producer channels pending'
        if qp or sp:
            obj=json.loads(q[qr] if qp else s[sr]);assert obj['seq']==seq,('episode gap/order',seq,obj['seq'])
            return ('QUERY' if qp else 'SKIP'),obj
        assert time.monotonic()-start<(600 if seq==0 else 30),'producer event timeout'
        if(d/'PRODUCER.log').exists():raise RuntimeError('producer stopped before expected episode')
        time.sleep(.005)

def run(a):
    assert a.episodes<=128;a.out.mkdir(parents=True,exist_ok=False)
    g=Layout('oct9');enc=Encoder('oct9');circuits={n:Circuit(g,[np.fromfile(a.prepared/f'bank{k}.ram','<u4') for k in range(2)]) for n in ARMS}
    for c in circuits.values():c.bootstrap()
    stores={n:Store(k) for n,k in [('exact32','exact'),('Hamming32','hamming'),('float32','float'),('FIFO32','fifo'),('LRU32','lru')]}
    qr=sr=0;atomic(a.out/'STATUS.json',dict(status='WAITING',physical=False,episodes=0,query_rows=0,skip_rows=0))
    try:
        for seq in range(a.episodes):
            kind,r=next_event(a.stream,seq,qr,sr);seen=time.monotonic_ns()
            if kind=='SKIP':
                assert set(r)=={'seq','skip_row','event_id','reason'} and r['skip_row']==sr and r['reason']=='observed terminal within probe'
                assert len(r['event_id'])==32
                append(a.stream/'SKIP_ACK.jsonl',dict(seq=seq,skip_row=sr,event_id=r['event_id']))
                append(a.out/'SKIPS.jsonl',dict(seq=seq,skip_row=sr,event_id=r['event_id'],seen_ns=seen,acked_ns=time.monotonic_ns(),taught=False));sr+=1
            else:
                assert r['query_row']==qr;q,wm,program=validate_query(r,seq)
                key,projection=enc.encode(q['h0_pool'],q['h5_pool'],program);raw=np.r_[q['h0_pool'],q['h5_pool'],np.subtract(q['h5_pool'],q['h0_pool'])]
                answer={n:c.query(key) for n,c in circuits.items()};answer.update({n:s.query(key,raw,program) for n,s in stores.items()});pred={n:fuse(wm,v) for n,v in answer.items()};pred['WM']=wm
                pc=int(pred['error-only'].argmax());append(a.stream/'PREDICTION_ACK.jsonl',dict(seq=seq,query_row=qr,event_id=r['event_id'],query_sha=r['query_sha'],arm='error-only',pred_class='DSC'[pc],pred_prob=float(pred['error-only'][pc]),mode='CPU_ENSEMBLE_N',native_receipt=None))
                acked=time.monotonic_ns();f=next_line(a.stream/'FEEDBACK.jsonl',qr);fb=time.monotonic_ns()
                assert f['seq']==seq and f['query_row']==f['release_index']==qr and f['event_id']==r['event_id'];truth=LABEL[f['outcome']];wc=int(wm.argmax())
                for name,c in circuits.items():c.feedback(key,wc,truth,admit_all=name=='all-admit',freeze=name=='after-A-frozen' and seq>=32)
                for s in stores.values():
                    s.retire_contradictions(key,raw,program,truth)
                    if wc!=truth:s.feedback(key,raw,program,truth)
                append(a.stream/'FEEDBACK_APPLIED.jsonl',dict(seq=seq,query_row=qr,event_id=r['event_id']));applied=time.monotonic_ns();assert seen<acked<fb<applied
                append(a.out/'RECORDS.jsonl',dict(seq=seq,query_row=qr,event_id=r['event_id'],query_sha=r['query_sha'],key_hex=key.tobytes().hex(),projection=projection.tolist(),truth=truth,wm_class=wc,answers=answer,predictions={n:v.tolist() for n,v in pred.items()},times_ns=[seen,acked,fb,applied]));qr+=1
            atomic(a.out/'STATUS.json',dict(status='RUNNING',physical=False,episodes=seq+1,query_rows=qr,skip_rows=sr))
        result=dict(status='N_CONSUMER_DONE_AWAITING_SCORE',physical=False,episodes=a.episodes,query_rows=qr,skip_rows=sr,all_skips_explicit_and_not_taught=True,no_AUDIT_or_plan_read=True)
        atomic(a.out/'STATUS.json',result);return result
    except BaseException as e:
        atomic(a.out/'STATUS.json',dict(status='FAILED_PRESERVED',physical=False,query_rows=qr,skip_rows=sr,error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--episodes',type=int,default=128);print(json.dumps(run(p.parse_args())))
