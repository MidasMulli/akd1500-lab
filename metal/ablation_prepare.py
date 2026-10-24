"""Freeze per-NPU threshold ablations and exact generated memory packets."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import akida
from metal.prepare import N, SEED, stream, mapping
from metal.prepare_cnp import Select

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'tools'))
from program_codec import read, words


def memory_packets(program):
    h, data = read(program); result = []
    assert len(h['passes']) == 1 and h['learning'] is None
    for record in h['passes'][0]:
        for span in record:
            a = words(data, span); at = 0
            while at < len(a):
                header, size = a[at:at+2]; length = (size>>16)&0x3fff
                if size>>31: length *= 16
                packet=a[at:at+2+length]; assert len(packet)==2+length
                mask, uid=(header>>8)&15, header&15
                if mask and uid in [0,2,4,6]:
                    assert not mask&(mask-1)
                    result.append(dict(np=[header>>24,(header>>16)&255,mask.bit_length()-1],packet=packet))
                at+=len(packet)
            assert at==len(a)
    return result


def main(study,attempt):
    out=study/attempt;out.mkdir(exist_ok=False)
    x,rows=stream();x=x[[i*16+j for i in range(7) for j in range(8)]+list(range(400,408))]
    rng=np.random.default_rng(SEED+19);u=256;w=np.zeros((N,u),np.int8)
    for j in range(u):w[rng.choice(N,32,replace=False),j]=1
    results=[];packets={}
    for treatment in ['baseline','suppress_A','suppress_B','suppress_both']:
        thresholds=np.full(u,15,np.int32)
        if treatment in ['suppress_A','suppress_both']:thresholds[:128]=32
        if treatment in ['suppress_B','suppress_both']:thresholds[128:]=32
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
        m.add(akida.Convolutional(name='partitioned_resident_work',filters=u,kernel_size=(1,1),weights_bits=1,activation=True,act_bits=1))
        l=m.get_layer('partitioned_resident_work');l.set_variable('weights',w.reshape(1,1,N,u));l.set_variable('threshold',thresholds);l.set_variable('act_step',np.ones(u,np.float32))
        m.add(akida.FullyConnected(name='two_support_outputs',units=2,weights_bits=1,activation=False))
        rw=np.zeros((u,2),np.int8);rw[:128,0]=1;rw[128:,1]=1
        m.get_layer('two_support_outputs').set_variable('weights',rw.reshape(1,1,u,2))
        response=x.astype(np.int32)@w.astype(np.int32)>thresholds
        expected=np.stack([response[:,:128].sum(axis=1),response[:,128:].sum(axis=1)],axis=1).astype('<i4')
        assert np.array_equal(m.forward(x.reshape(-1,1,1,N)).reshape(-1,2),expected)
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==1 and len(m.sequences[0].passes)==1
        program=bytes(m.sequences[0].program);p=memory_packets(program)
        assert len(p)==5 and len({tuple(t['np']) for t in p})==3, [(t['np'],len(t['packet'])) for t in p]
        packets[treatment]=p;f=out/treatment
        for ext,data in [('bin',program),('inputs',x.tobytes()),('expected',expected.tobytes())]:f.with_suffix('.'+ext).write_bytes(data)
        layout=[f'{N} 2 {len(x)} {len(p)}']
        for k,packet in enumerate(p):
            a=packet['packet'];(out/f'{treatment}.packet{k}.ram').write_bytes(np.asarray(a,'<u4').tobytes())
            layout.append(' '.join(map(str,packet['np']+[len(a)])))
        f.with_suffix('.layout').write_text('\n'.join(layout)+'\n')
        meta=dict(name=treatment,mapping=mapping(m),thresholds=thresholds.tolist(),
                  packets=[dict(np=t['np'],words=len(t['packet']),sha256=hashlib.sha256(np.asarray(t['packet'],'<u4').tobytes()).hexdigest()) for t in p],
                  program_sha256=hashlib.sha256(program).hexdigest())
        f.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');results.append(meta)
    changes={}
    baseline=packets['baseline']
    for treatment,other in packets.items():
        assert [q['np'] for q in baseline]==[q['np'] for q in other]
        changed=[]
        for a,b in zip(baseline,other):
            assert len(a['packet'])==len(b['packet']) and a['packet'][:2]==b['packet'][:2]
            ids=[i for i,(v,z) in enumerate(zip(a['packet'],b['packet'])) if v!=z]
            if ids:changed.append(dict(np=a['np'],changed_words=ids))
        changes[treatment]=changed
    assert len(changes['suppress_A'])==len(changes['suppress_B'])==1
    assert changes['suppress_A'][0]['np']!=changes['suppress_B'][0]['np']
    assert len(changes['suppress_both'])==2
    report=dict(status='VIRTUAL_AND_PACKET_LOCALIZATION_PASS',physical_device_opened=False,fixtures=results,packet_changes=changes)
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],changes=changes)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--attempt',default='fixtures-ablation02');a=p.parse_args();main(a.study,a.attempt)
