"""Causal GPU-query consumer. CPU ensemble first; one resident native arm later.

The acting path never opens PLAN, AUDIT or CANDIDATES. Native feedback operands
are routed by the actor's descriptors, with an independent CPU shadow for QA.
"""
import argparse, datetime, hashlib, json, os, select, struct, subprocess, time
from pathlib import Path
import numpy as np
from core import Circuit, Layout
from encoder import Encoder
from fixture import Recorder

LABEL = {'D': 0, 'S': 1, 'C': 2}
ARMS = ['error-only', 'all-admit', 'S1AB-frozen']
DEADLINE = datetime.datetime(2026, 10, 28, 11, 55, tzinfo=datetime.timezone.utc)

def sha_bytes(x): return hashlib.sha256(x).hexdigest()
def digest(x): return sha_bytes(json.dumps(x, sort_keys=True).encode())
def check_deadline(): assert datetime.datetime.now(datetime.timezone.utc) < DEADLINE, 'scientific cutoff'
def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
def atomic(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)

def next_line(path, index, timeout=30):
    start = time.monotonic()
    while True:
        check_deadline()
        lines = path.read_text().splitlines() if path.exists() else []
        assert len(lines) <= index + 1, ('duplicate or extra line', str(path), index)
        if len(lines) == index + 1:
            return json.loads(lines[index])
        assert time.monotonic() - start < timeout, ('consumer timeout', str(path), index)
        time.sleep(.005)

def validate_query(row, seq):
    assert row['seq'] == seq and len(row['event_id']) == 32
    q = row['query']; assert digest(q) == row['query_sha']
    assert set(q) == {'h0_pool','h5_pool','wm_logits','wm_p','wm_pT','program_id','program_ux','probe_ux_cmd'}
    for key, shape in [('h0_pool',(128,)),('h5_pool',(128,)),('wm_logits',(35,3)),('wm_p',(3,)),('wm_pT',(3,)),('program_ux',(40,)),('probe_ux_cmd',(5,))]:
        a = np.asarray(q[key], 'f8'); assert a.shape == shape and np.isfinite(a).all(), key
    for name in ['wm_p','wm_pT']:
        p = np.asarray(q[name]); assert np.all(p >= 0) and abs(p.sum()-1) < 1e-6
    lg = np.asarray(q['wm_logits'], 'f8') / 1.58
    ex = np.exp(lg-lg.max(1,keepdims=True)); h = ex/ex.sum(1,keepdims=True)
    survive = np.r_[1.,np.cumprod(h[:,0])]
    expected = np.array([survive[-1], (survive[:-1]*h[:,1]).sum(), (survive[:-1]*h[:,2]).sum()])
    wm = np.asarray(q['wm_pT'], 'f8')[[2,0,1]]  # Producer S,C,D -> consumer D,S,C, prospectively frozen.
    assert np.allclose(wm,expected,atol=2e-6,rtol=0), 'probability composition/order'
    program = int(q['program_id'][1:]); assert program in [0,1,4]
    ux = {0:1.,1:.8,4:.6}[program]
    assert np.all(np.asarray(q['program_ux']) == ux) and np.all(np.asarray(q['probe_ux_cmd']) == ux)
    return q, wm, program

def pack(key): return np.packbits(key,bitorder='little').tobytes()
class Store:
    """Task-level CPU controls, same 32-record budget and error-only admission."""
    def __init__(self, kind): self.kind=kind; self.items=[]
    def query(self,key,raw,program):
        p=pack(key); eligible=[v for v in self.items if v[2]==program]
        if self.kind=='float': return min(reversed(eligible),key=lambda v:np.square(v[3]-raw).sum())[1] if eligible else None
        if self.kind=='hamming': return min(reversed(eligible),key=lambda v:sum((a^b).bit_count() for a,b in zip(v[0],p)))[1] if eligible else None
        for i,v in enumerate(self.items):
            if v[0]==p:
                if self.kind=='lru': self.items.append(self.items.pop(i))
                return v[1]
        return None
    def feedback(self,key,raw,program,truth):
        p=pack(key)
        same=[i for i,v in enumerate(self.items) if v[2]==program and (np.array_equal(v[3],raw) if self.kind=='float' else v[0]==p)]
        if same:
            i=same[0]; self.items[i]=(p,truth,program,raw.copy())
            if self.kind in ['lru','float','hamming']: self.items.append(self.items.pop(i))
        elif len(self.items)<32 or self.kind!='exact':
            self.items.append((p,truth,program,raw.copy())); self.items=self.items[-32:]
    def retire_contradictions(self,key,raw,program,truth):
        p=pack(key)
        # Mirrors the native wrong-answer retirement on every observation.
        # The stronger full-float control can distinguish states that alias in
        # the binary encoding; it retires only an identical full-float state.
        def same(v):
            return v[2]==program and (np.array_equal(v[3],raw) if self.kind=='float' else v[0]==p)
        self.items=[v for v in self.items if not (same(v) and v[1]!=truth)]

def fuse(wm, value): return wm.copy() if value is None else .25*wm+.75*np.eye(3)[value]

class ActorRecorder(Recorder):
    def __init__(self, prepared, out, actor, mode, forwards):
        super().__init__(prepared,out,'oct9'); self.prefix=out/'RAW'; self.error=(out/'ACTOR.stderr').open('wb')
        self.actor=subprocess.Popen([str(actor),mode,str(out),str(self.prefix),str(forwards)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error)
    def add(self,net,x,mode=-1,label=0,external=False,candidate=-1,phase=-1):
        y=super().add(net,x,mode,label,external,candidate,phase)
        padded=np.zeros(1024,'u1'); padded[:len(x)]=x
        self.actor.stdin.write(struct.pack('<iiii',net,mode,label,int(external))+padded.tobytes()); self.actor.stdin.flush()
        data=bytearray(); start=time.monotonic()
        while len(data)<4*self.g.W:
            check_deadline(); remaining=15-(time.monotonic()-start); assert remaining>0,'actor response timeout'
            ready,_,_=select.select([self.actor.stdout],[],[],remaining); assert ready,'actor response timeout'
            part=os.read(self.actor.stdout.fileno(),4*self.g.W-len(data)); assert part,('actor exited',self.actor.poll())
            data.extend(part)
        actual=np.frombuffer(data,'<u4'); assert np.array_equal(actual,self.outputs[-1]),'actual actor output differs from Python shadow'
        return actual[:len(y)].copy()
    def sync_raw(self):
        for p in self.out.glob('RAW.*'):
            if p.is_file():
                with p.open('rb') as f: os.fsync(f.fileno())
    def finish(self):
        self.actor.stdin.close(); tail=self.actor.stdout.read(); assert self.actor.wait(timeout=10)==0,tail
        assert json.loads(tail)['status']=='PASS'; self.error.close()
    def abort(self):
        if self.actor.poll() is None:
            self.actor.terminate()
            try:self.actor.wait(timeout=5)
            except subprocess.TimeoutExpired:self.actor.kill();self.actor.wait()
        self.error.close()

def run(a):
    assert a.K in [8,16,32] and 0<a.events<=288
    a.out.mkdir(parents=True,exist_ok=False)
    atomic(a.out/'STATUS.json',dict(status='WAITING_FOR_FIRST_QUERY',physical=False,events=0))
    # This is startup coordination, before QUERY exists. Once QUERY is emitted,
    # the producer still enforces the frozen 30-second acknowledgement clock.
    startup=time.monotonic()
    while not (a.stream/'QUERY.jsonl').exists():
        check_deadline(); assert time.monotonic()-startup<180,'producer startup timeout'
        time.sleep(.02)
    enc=Encoder('oct9'); g=Layout('oct9')
    actor=None
    if a.mode=='ensemble':
        circuits={name:Circuit(g,[np.fromfile(a.prepared/f'bank{k}.ram','<u4') for k in range(2)]) for name in ARMS}
        for c in circuits.values():c.bootstrap()
    else:
        forwards=36+54*a.events; actor=ActorRecorder(a.prepared,a.out/'fixture',a.actor,a.mode,forwards); actor.bootstrap(); circuits={a.arm:actor}
    stores={name:Store(kind) for name,kind in [('exact32','exact'),('Hamming32','hamming'),('float32','float'),('FIFO32','fifo'),('LRU32','lru')]}
    records=[]
    try:
        for seq in range(a.events):
            check_deadline(); row=next_line(a.stream/'QUERY.jsonl',seq); seen=time.monotonic_ns(); q,wm,program=validate_query(row,seq)
            key,projection=enc.encode(q['h0_pool'],q['h5_pool'],program); raw=np.r_[q['h0_pool'],q['h5_pool'],np.subtract(q['h5_pool'],q['h0_pool'])]
            if actor: actor.episode=seq
            answers={name:c.query(key) for name,c in circuits.items()}; answers={name:None if v==-1 else v for name,v in answers.items()}
            answers.update({name:s.query(key,raw,program) for name,s in stores.items()})
            predictions={name:fuse(wm,v) for name,v in answers.items()}; predictions['WM']=wm
            selected=a.arm if actor else 'error-only'; p=predictions[selected]; pc=int(p.argmax())
            if actor:actor.sync_raw()
            append(a.stream/'PREDICTION_ACK.jsonl',dict(seq=seq,event_id=row['event_id'],query_sha=row['query_sha'],arm=selected,pred_class='DSC'[pc],pred_prob=float(p[pc]),pred_vector=p.tolist(),mode=a.mode,native_receipt=dict(physical=a.mode=='stream-native',last_completed_forward=len(actor.rows)-1 if actor else None)))
            acked=time.monotonic_ns(); fb=next_line(a.stream/'FEEDBACK.jsonl',seq); feedback_seen=time.monotonic_ns()
            assert fb['seq']==seq and fb['event_id']==row['event_id'] and fb['release_index']==seq and fb['outcome'] in LABEL
            truth=LABEL[fb['outcome']]; wc=int(wm.argmax())
            for name,c in circuits.items():
                freeze=name=='S1AB-frozen' and seq>=3*a.K
                if actor:c.feedback(key,wc,truth,all_admit=name=='all-admit',freeze=freeze)
                else:c.feedback(key,wc,truth,admit_all=name=='all-admit',freeze=freeze)
            for s in stores.values():
                s.retire_contradictions(key,raw,program,truth)
                if wc!=truth:s.feedback(key,raw,program,truth)
            if actor:actor.sync_raw()
            append(a.stream/'FEEDBACK_APPLIED.jsonl',dict(seq=seq,event_id=row['event_id'],query_sha=row['query_sha']))
            applied=time.monotonic_ns()
            assert seen<acked<feedback_seen<applied
            rec=dict(seq=seq,event_id=row['event_id'],query_sha=row['query_sha'],key_hex=key.tobytes().hex(),projection=projection.tolist(),truth=truth,wm_class=wc,answers=answers,predictions={k:v.tolist() for k,v in predictions.items()},times_ns=[seen,acked,feedback_seen,applied])
            append(a.out/'RECORDS.jsonl',rec);records.append(rec)
            if actor:actor.answers.append(dict(episode=seq,query=answers[a.arm] if answers[a.arm] is not None else -1,truth=truth,wm_class=wc))
            atomic(a.out/'STATUS.json',dict(status='RUNNING',physical=a.mode=='stream-native',events=seq+1,requested=a.events))
        if actor:
            actor.finish(); actor.save(dict(stream=True,mode=a.mode,K=a.K,arm=a.arm,events=a.events))
        result=dict(status='CONSUMER_DONE_AWAITING_SCORE',physical=a.mode=='stream-native',events=len(records),K=a.K,mode=a.mode,query_before_current_outcome_release=True,no_audit_or_plan_read_by_acting_path=True,probability_adapter=[2,0,1])
        atomic(a.out/'STATUS.json',result); return result
    except BaseException as e:
        if actor:actor.abort()
        atomic(a.out/'STATUS.json',dict(status='FAILED_PRESERVED',events=len(records),error=repr(e),physical=a.mode=='stream-native'));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--events',type=int,required=True);p.add_argument('--K',type=int,required=True);p.add_argument('--mode',choices=['ensemble','stream-cpu','stream-native'],default='ensemble');p.add_argument('--actor',type=Path);p.add_argument('--arm',choices=ARMS,default='error-only');print(json.dumps(run(p.parse_args())))
