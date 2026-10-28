"""Offline audit of saved resident trials; no device/SDK required."""
import argparse
import json
from pathlib import Path
import numpy as np
from instruments.explorer.data import sha
from instruments.explorer.live import Stepper
from admission import learned_policy as model
from admission import learned_policy_run as run


def audit(path):
    d=Path(path);r=json.loads((d/'RECEIPT.json').read_text())
    assert r['status']=='CLOSED_VERIFIED' and r['exit_code']==0 and r['device_free'] and r['cleanup_acknowledged']
    for name,digest in r['hashes'].items():assert sha(d/name)==digest,('capture hash',name)
    rows=[json.loads(line) for line in (d/'events.jsonl').read_text().splitlines()]
    shown=[json.loads(line) for line in (d/'verified.jsonl').read_text().splitlines()]
    n=len(rows);assert n==r['events']==len(shown)
    s=Stepper(run.fixture(151101,r['arm']),r['arm']);prefix=d/'result'
    def a(field,dtype,width):
        x=np.fromfile(str(prefix)+'.'+field,dtype);assert x.size==n*width,(field,x.size);return x.reshape(n,width)
    xx=a('inputs','u1',128);yy=a('scores','u1',128);modes=a('learning_modes','<i4',3)
    states=[a(f'states{k}','<u4',len(s.initial[k])) for k in range(3)]
    previous=None;query_count=0
    if r['mode']=='native':
        desc=a('descriptors','<u4',16);inp=a('input_addresses','<u4',1).ravel();out=a('output_addresses','<u4',1).ravel()
        assert np.array_equal(desc[:,0]>>16,np.arange(1,n+1))
        assert np.all((desc[:,0]&65535)==0x2001) and np.all(desc[:,2]==0x10080)
        assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],out)
        assert not desc[:,[3,4,5,6,9,10,11,12,13,14,15]].any()
        assert not (desc[:,8]&65535).any()
        for k in range(3):assert np.array_equal(np.fromfile(str(prefix)+f'.initial{k}','<u4'),s.initial[k])
        for k in range(4):
            fixture=np.fromfile(s.fixture/f'fixed{k}.ram','<u4')
            for tag in ['initial','final']:assert np.array_equal(np.fromfile(str(prefix)+f'.fixed{k}.{tag}','<u4'),fixture)
    for i,e in enumerate(rows):
        expected,x,y,ram=s.forward(e)
        assert np.array_equal(x,xx[i]) and np.array_equal(y,yy[i]),('native input/output',i)
        assert shown[i]==dict(expected,evidence_mode='captured_native' if r['mode']=='native' else 'cpu_only')
        assert modes[i].tolist()==[int(v>=0) for v in e['labels']]
        for k in range(3):assert np.array_equal(states[k][i],ram[k]),('RAM',i,k)
        if e['role'] in ['exam','query']:
            assert not expected['changes'];query_count+=1
        if r['mode']=='native':
            assert inp[i]!=out[i]+32
            if previous is not None:assert inp[i]==out[previous]+32,('native SRAM carry',i)
            packet=np.fromfile(str(prefix)+f'.packet{i}','<u4');assert len(packet)==40
            assert (int(packet[0])&0x1fffffff)==32
            assert np.array_equal(packet.view('u1')[32:],y)
            assert ((int(desc[i,8])>>16)&1023)==e['wire']+1
            if e['role']!='transfer':previous=i
    return dict(status='OFFLINE_RESIDENT_AUDIT_PASS',path=str(d),arm=r['arm'],mode=r['mode'],
                events=n,query_only_boundaries=query_count,packet_and_native_carry_verified=r['mode']=='native',
                receipt_sha256=sha(d/'RECEIPT.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
    print(json.dumps([audit(path) for path in a.paths],indent=2))
