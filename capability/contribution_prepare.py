"""Prospective differential compiler localization of every large-graph CNP."""
import json
import numpy as np
import akida
from capability.design import P,weights,inputs
from metal.prepare import mapping
from metal.prepare_cnp import Select
from metal.ablation_prepare import memory_packets
from tools.prepare_network import fc
from metal.run import digest
S=P/'studies/20261024-cnp-contribution'


def main():
    out=S/'fixtures01';out.mkdir(exist_ok=False);parent=P/'studies/20261024-native-primitives/fixtures01/cnp2048.json';info=json.loads(parent.read_text());packets=info['packets']
    ps=[p for p in packets if p['np'] in [q['np'] for q in info['mapping'][0]['nps']] and p['words']!=8194];counts=[(p['words']-2)//2 for p in ps];assert len(counts)==11 and sum(counts)==2048
    x,rows=inputs();w=weights();ends=np.cumsum([0]+counts);all_packets={};result=[]
    for t in range(-1,11):
        name='baseline' if t<0 else f'suppress{t}';th=np.full(2048,15,np.int32)
        if t>=0:th[ends[t]:ends[t+1]]=32
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,512),input_bits=1));m.add(akida.Convolutional(name='binary_matches',filters=2048,kernel_size=(1,1),weights_bits=1,activation=True,act_bits=1));l=m.get_layer('binary_matches');l.set_variable('weights',w.reshape(1,1,512,2048).astype(np.int8));l.set_variable('threshold',th);l.set_variable('act_step',np.ones(2048,np.float32));rw=np.zeros((2048,2),np.int8);rw[:1024,0]=1;rw[1024:,1]=1;fc(m,'two_counts',rw,None,activation=False,bits=1)
        y=(x.astype(np.int32)@w.astype(np.int32)>th).reshape(160,2,1024).sum(2).astype('<i4');assert np.array_equal(y,m.forward(x.reshape(-1,1,1,512)).reshape(-1,2))
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1;mp=mapping(m);assert mp==info['mapping'];program=bytes(m.sequences[0].program);pk=memory_packets(program)
        if t<0:
            assert digest(parent.parent/'cnp2048.bin')==__import__('hashlib').sha256(program).hexdigest(),'identical parent baseline program';base=pk;changed=[]
        else:
            assert len(pk)==len(base);changed=[]
            for j,(a,b) in enumerate(zip(base,pk)):
                assert a['np']==b['np'] and len(a['packet'])==len(b['packet']) and a['packet'][:2]==b['packet'][:2]
                ii=[i for i,(aa,bb) in enumerate(zip(a['packet'],b['packet'])) if aa!=bb]
                if ii:changed.append(dict(packet=j,np=a['np'],changed_words=ii))
            assert len(changed)==1 and changed[0]['np']==ps[t]['np'],'single-CNP threshold localization'
        f=out/name;f.with_suffix('.bin').write_bytes(program);f.with_suffix('.inputs').write_bytes(x.tobytes());f.with_suffix('.expected').write_bytes(y.tobytes());f.with_suffix('.thresholds').write_bytes(th.astype('<i4').tobytes());f.with_suffix('.meta').write_text('512 0 0 2\n');f.with_suffix('.regions').write_text('\n'.join([str(len(pk))]+[str(len(p['packet'])) for p in pk])+'\n')
        for j,p in enumerate(pk):(out/f'{name}.packet{j}.ram').write_bytes(np.asarray(p['packet'],'<u4').tobytes())
        rec=dict(name=name,kind='contribution',units=2048,status='COMPILER_LOCALIZATION_PASS',mapping=mp,thresholds=th.tolist(),suppressed_block=None if t<0 else [int(ends[t]),int(ends[t+1])],changed_packets=changed,packets=[dict(np=p['np'],words=len(p['packet']),sha256=digest(out/f'{name}.packet{j}.ram')) for j,p in enumerate(pk)],program_sha256=digest(f.with_suffix('.bin')))
        f.with_suffix('.json').write_text(json.dumps(rec,indent=2)+'\n');result.append(rec);(out/'BUILD.json').write_text(json.dumps(dict(status='COMPILER_LOCALIZATION_PASS',physical_device_opened=False,counts=counts,cells=result),indent=2)+'\n');print(json.dumps(dict(name=name,status=rec['status'],changed=changed)),flush=True)


if __name__=='__main__':main()
