"""Bounded live-worker protocol; exact stepwise Python/packed/native comparison.

Runs on the Nano over a private SSH stdio channel. No network listener on Nano,
arbitrary paths, remote shell text, or reads of a different owner's device state.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import uuid
import numpy as np
from admission import learned_policy as model
from admission import learned_policy_run as run
from admission.recycle import Reference
from instruments.explorer.data import records, sha

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'studies/20261025-network-explorer/runs'


def validate_action(a):
    if not isinstance(a,dict): raise ValueError('object required')
    action=a.get('action')
    fields={'start':{'action','arm','mode'},'query':{'action','key'},
            'propose':{'action','key','policy','candidate'},
            'feedback':{'action','policy','candidate','outcome'},'stop':{'action'}}
    if action not in fields or set(a)!=fields[action]: raise ValueError('exact action fields required')
    for k in ['key','policy','candidate','outcome']:
        if k in a and (type(a[k]) is not int or not 0<=a[k]<({'key':6,'policy':5}.get(k,2))):
            raise ValueError('bounded integer '+k)
    if action=='feedback' and a['policy']==4: raise ValueError('unknown source has no feedback identity')
    if action=='start' and (a['arm'] not in model.ARMS or a['mode'] not in ['native','cpu']):
        raise ValueError('known arm/mode required')
    return a


def action_rows(a, arm, index):
    validate_action(a); action=a['action']; rows=[]
    if action in ['start','stop']:return rows
    p=a.get('policy',0); cand=a.get('candidate',0);outcome=a.get('outcome')
    roles=['exam'] if action=='query' else ['query','transfer']*3 if action=='propose' else ['feedback','feedback_negative']*3
    for role in roles:
        good=int(cand==outcome) if outcome is not None else -1
        wire=(1-good if role=='feedback_negative' else good) if role.startswith('feedback') else 0
        labels=[wire,-1,-1] if role.startswith('feedback') else [-1,0,0] if role=='transfer' else [-1,-1,-1]
        if arm=='manager_frozen':labels[0]=-1
        if arm=='students_frozen':labels[1:]=[-1,-1]
        rows.append(dict(index=index+len(rows),key=8+p if action=='feedback' else a['key'],
                         policy_key=p,candidate=cand,role=role,phase='interactive_'+action,block=1,
                         actual=0,outcome=outcome,quality=good,wire=wire,labels=labels,
                         expected_permit=None,ground_truth_supplied=False))
    return rows


class Stepper:
    def __init__(self, fixture, arm):
        self.arm=arm;self.fixture=Path(fixture);self.rows=[];self.last=np.zeros(128,'u1')
        self.initial=[np.fromfile(self.fixture/f'bank{k}.ram','<u4') for k in range(3)]
        self.refs=[Reference(r,n=512,u=dim[0],nw=32,classes=2) for r,dim in zip(self.initial,model.DIMENSIONS)]
        self.mats=model.matrices(arm)
        self.before=[records(r,dim[0]) for r,dim in zip(self.initial,model.DIMENSIONS)]

    def forward(self,e):
        x=model.observation(self.last,e)
        z=(x.astype('i4')@self.mats[0][0].astype('i4')>self.mats[0][1]).astype('u1')
        features=[];potentials=[];changes=[];counts=[];states=[]
        for k,ref in enumerate(self.refs):
            features.append(np.flatnonzero(z).tolist())
            p=np.asarray(ref.forward(z,e['labels'][k]),'<i4');potentials.append(p.tolist())
            z=(p>model.thresholds(k)[1]).astype('u1');ram=np.asarray(ref.ram,'<u4');states.append(ram)
            after=records(ram,model.DIMENSIONS[k][0]);added=removed=0
            for j,rec in enumerate(after):
                old=self.before[k][j]
                if rec!=old:
                    aa=sorted(set(rec['connections'])-set(old['connections']));rr=sorted(set(old['connections'])-set(rec['connections']))
                    changes.append(dict(bank=k,record=j,before=old,after=rec,added=aa,removed=rr));added+=len(aa);removed+=len(rr)
            counts.append(dict(added=added,removed=removed));self.before[k]=after
            if k<2:
                for w,t in self.mats[1:3] if k==0 else self.mats[3:4]:z=(z.astype('i4')@w.astype('i4')>t).astype('u1')
        if e['role']!='transfer':self.last=z.copy()
        self.rows.append(e)
        names,cpu=model.cpu_controls(self.rows)
        event=dict(**e,policy_flags=z[84:86].tolist(),memory_flags=[int(z[88]),int(z[model.PR].any())],
                   output_active=np.flatnonzero(z).tolist(),input_active=np.flatnonzero(x).tolist(),
                   changes=changes,counts=counts,derived_features=features,
                   derived_potentials=potentials,cpu_memory=cpu[-1].tolist())
        return event,x,z,states


class Resident:
    def __init__(self,mode,arm):
        self.mode=mode;self.arm=arm;self.lock=None;self.proc=None;self.closed=False;self.started=time.monotonic();self.buffer=b''
        self.fixture=run.fixture(151101,arm);self.stepper=Stepper(self.fixture,arm)
        original=run.RAW/run.name('native',151101,arm,'pilot',0)
        original_receipt=json.loads((original/'RECEIPT.json').read_text())
        if sha(original/'FROZEN.json')!=original_receipt['freeze_sha256']:raise RuntimeError('original fixture freeze changed')
        freeze=json.loads((original/'FROZEN.json').read_text())
        prefix=str(self.fixture.relative_to(ROOT))+'/'
        frozen_fixture={k:v for k,v in freeze['hashes'].items() if k.startswith(prefix)}
        if not frozen_fixture or prefix+'model.bin' not in frozen_fixture:raise RuntimeError('fixture not in frozen study')
        for name,digest in frozen_fixture.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('frozen fixture changed: '+name)
        self.path=RUNS/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True,exist_ok=False)
        self.manifest=dict(mode=mode,arm=arm,seed=151101,initialization='Fresh seeded program; bootstrap replayed once, then resident',
                           start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           bounds=dict(maximum_events=512,maximum_seconds=150,idle_seconds=45),
                           executable_sha256=sha(ROOT/'build/explorer_native'),
                           original_fixture_freeze_sha256=sha(original/'FROZEN.json'),
                           sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'instruments/explorer/native.cpp']},
                           fixtures={p.name:sha(p) for p in self.fixture.iterdir() if p.is_file()},
                           status='STARTING')
        self.save()
        try:
            if mode=='native':
                self.lock=open('/tmp/akd1500-research.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode!=1:raise RuntimeError('device occupied')
            self.err=(self.path/'stderr').open('w')
            self.proc=subprocess.Popen([str(ROOT/'build/explorer_native'),mode,str(self.fixture),str(self.path/'result')],
                                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,bufsize=0,cwd=ROOT)
            self.manifest['pid']=self.proc.pid;self.save();r=self.read(12)
            if r.get('type')!='ready':raise RuntimeError('native worker did not become ready')
            self.manifest['status']='RESIDENT';self.save()
        except BaseException:
            self.close(failed=True);raise

    def save(self):
        (self.path/'RECEIPT.json').write_text(json.dumps(self.manifest,indent=2)+'\n')

    def read(self,timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if b'\n' not in self.buffer:
                ready,_,_=select.select([self.proc.stdout],[],[],max(0,deadline-time.monotonic()))
                if not ready:break
                block=os.read(self.proc.stdout.fileno(),1048576)
                if not block:raise RuntimeError('worker ended: '+(self.path/'stderr').read_text()[-1500:])
                self.buffer+=block
            while b'\n' in self.buffer:
                raw,self.buffer=self.buffer.split(b'\n',1);line=raw.decode()
                with (self.path/'stdout').open('a') as f:f.write(line+'\n')
                if line.startswith('EXPLORER '):return json.loads(line[9:])
        raise TimeoutError('bounded native response timeout')

    def step(self,e):
        if self.closed or len(self.stepper.rows)>=512 or time.monotonic()-self.started>145:raise RuntimeError('session bound reached')
        row=[e['key'],model.ROLES.index(e['role']),*e['labels'],e['wire'],e['block'],e['policy_key'],e['candidate']]
        with (self.path/'events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
        self.proc.stdin.write((' '.join(map(str,row))+'\n').encode());self.proc.stdin.flush()
        marker=self.read(8)
        if marker.get('type')!='event' or marker.get('count')!=len(self.stepper.rows)+1:raise RuntimeError('event sequence mismatch')
        event,x,y,states=self.stepper.forward(e);n=len(self.stepper.rows)
        prefix=self.path/'result'
        def read(field,dtype,width):
            arr=np.fromfile(str(prefix)+'.'+field,dtype)
            if arr.size!=n*width:raise RuntimeError('incomplete capture '+field)
            return arr.reshape(n,width)[-1]
        assert np.array_equal(read('inputs','u1',128),x),'Python input mismatch'
        assert np.array_equal(read('scores','u1',128),y),'Python output mismatch'
        for k,state in enumerate(states):
            assert np.array_equal(read(f'states{k}','<u4',len(state)),state),'Python RAM mismatch'
            assert np.array_equal(read(f'potentials{k}','<i4',len(event['derived_potentials'][k])),event['derived_potentials'][k])
        if e['role']=='exam':assert not event['changes'],'read-only query modified memory'
        event['evidence_mode']='captured_native' if self.mode=='native' else 'cpu_only'
        with (self.path/'verified.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        return event

    def close(self,failed=False):
        if self.closed:return
        self.closed=True;cleanup=False
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write(b'STOP\n');self.proc.stdin.flush()
                    marker=self.read(5);cleanup=marker.get('type')=='closed'
                self.proc.wait(timeout=3)
            except Exception:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.err.close()
        free=self.mode!='native' or subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        if self.lock:self.lock.close()
        self.manifest.update(status='FAILED_PRESERVED' if failed or not cleanup or not free else 'CLOSED_VERIFIED',
                             events=len(self.stepper.rows),exit_code=self.proc.returncode if self.proc else None,
                             cleanup_acknowledged=cleanup,device_free=free,
                             finish_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                             hashes={p.name:sha(p) for p in self.path.iterdir() if p.is_file() and p.name!='RECEIPT.json'})
        self.save()


def worker():
    resident=None
    def emit(v):print(json.dumps(v,separators=(',',':')),flush=True)
    try:
        while True:
            ready,_,_=select.select([sys.stdin],[],[],35 if resident else 120)
            if not ready:break
            line=sys.stdin.readline(4097)
            if not line:break
            if len(line)>4096:raise ValueError('command too long')
            a=validate_action(json.loads(line))
            if a['action']=='start':
                if resident:raise ValueError('stop current session first')
                if a['mode']=='native':
                    gate=json.loads((Path(__file__).parent/'generated/NATIVE_QUALIFICATION.json').read_text())
                    if gate['status']!='NATIVE_QUERY_TEACH_PASS' or gate['native_binary_sha256']!=sha(ROOT/'build/explorer_native'):
                        raise RuntimeError('native binary is not qualified')
                    for name,digest in gate['source_hashes'].items():
                        if sha(Path(__file__).parent/name)!=digest:raise RuntimeError('qualified source changed')
                resident=Resident(a['mode'],a['arm'])
                emit(dict(type='started',mode=resident.mode,arm=resident.arm,path=str(resident.path),
                          initial=resident.stepper.before,cpu_names=model.cpu_controls([])[0],bounds=resident.manifest['bounds']))
                for e in model.schedule('support',resident.arm,0):emit(dict(type='event',event=resident.step(e)))
                emit(dict(type='ready',events=84))
            elif a['action']=='stop':
                if resident:resident.close();emit(dict(type='stopped',receipt=resident.manifest));resident=None
                else:emit(dict(type='stopped',receipt=None))
            else:
                if not resident:raise ValueError('start a session first')
                rows=action_rows(a,resident.arm,len(resident.stepper.rows))
                if len(resident.stepper.rows)+len(rows)>512:raise ValueError('event budget')
                for e in rows:emit(dict(type='event',event=resident.step(e)))
                emit(dict(type='ready',events=len(resident.stepper.rows)))
    except BaseException as error:
        if resident:resident.close(failed=True)
        emit(dict(type='error',error=str(error)));raise
    finally:
        if resident and not resident.closed:resident.close();emit(dict(type='stopped',receipt=resident.manifest))


if __name__=='__main__':worker()
