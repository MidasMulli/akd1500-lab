"""Compiler-only, single resident six-group memory-management circuit."""
from pathlib import Path
import argparse,json,struct,sys,traceback,hashlib
import numpy as np
import akida
from admission.model import *
from tools.program_codec import read,build,words
from tools.prepare import parts
from metal.prepare import mapping
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select

def fixed(m,name,w,t):
    m.add(akida.FullyConnected(name=name,units=w.shape[1],weights_bits=1,activation=True,act_bits=1))
    l=m.get_layer(name);l.set_variable('weights',w.reshape(1,1,*w.shape));l.set_variable('threshold',t);l.set_variable('act_step',np.ones(w.shape[1],np.float32))

def bind(prefixes,app,dimensions=None):
    image,adapter=assemble(prefixes[-1],app,N1,U1)
    h,data=read(image);data=bytearray(data);banks=[]
    for k,prefix in enumerate(prefixes):
        hh,dd=read(prefix);lr=hh['learning'];ram=words(dd,lr['ram'][0]);inf=words(dd,lr['inference']);lea=words(dd,lr['learning'])
        n,u=(dimensions or [(N0,U0),(N1,U1)])[k];stride=4+n//32
        if k==len(prefixes)-1:ni=words(data,h['learning']['inference']);nl=words(data,h['learning']['learning'])
        else:
            candidates=[(j,words(data,sp)) for j,rec in enumerate(h['passes'][0]) for sp in rec if len(words(data,sp))==52+len(ram) and words(data,sp)[0]==inf[0] and words(data,sp)[52:54]==ram[:2]]
            assert len(candidates)==1
            ri,packet=candidates[0];ni=packet[:52];nl=ni[:]
            assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b)
            assert all(j>=2 and (j-2)%stride in [1,2,3] for j,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b)
            for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
            sp=[len(data),52+len(ram)];data.extend(np.asarray(ni+ram,'<u4').tobytes());h['passes'][0][ri]=[sp]
        banks.append(dict(np=lr['np'],ram=ram,infer=ni,learn=nl,n=n,u=u))
    struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));assert read(image)==(h,bytes(data));fixed_packets=[]
    for rec in h['passes'][0]:
        for span in rec:
            a=words(data,span);at=0
            while at<len(a):
                header,size=a[at:at+2];length=(size>>16)&0x3fff
                if size>>31:length*=16
                packet=a[at:at+2+length];assert len(packet)==2+length;at+=len(packet);mask,uid=(header>>8)&15,header&15
                if mask and uid in [0,2,4,6]:
                    assert not mask&(mask-1);npid=[header>>24,(header>>16)&255,mask.bit_length()-1]
                    if npid not in [b['np'] for b in banks]:fixed_packets.append(dict(np=npid,packet=packet))
            assert at==len(a)
    assert len(fixed_packets)==len({tuple(p['np']) for p in fixed_packets})==4
    return image,banks,fixed_packets,adapter

def main(out,seed,arm):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,arm=arm,physical_device_opened=False)
    try:
        mats=matrices(arm);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));prefixes=[]
        fixed(m,'raw_current_and_native_history',*mats[0]);fixed(m,'native_admission_policy',*mats[1])
        for k,(n,u,plastic) in enumerate([(N0,U0,P0),(N1,U1,P1)]):
            if k:fixed(m,'native_bank_bridge',*mats[2])
            m.add(akida.FullyConnected(name=f'plastic_bank_{k}_with_fixed_helpers',units=u,weights_bits=1,activation=True,act_bits=1))
            m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=2,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
            l=m.layers[-1];w=weights(seed,k);lt=np.full(u,32,'i4');lt[plastic]=0;t=np.full(u,31,'i4');t[plastic]=27
            l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',lt);l.set_variable('threshold',t);l.set_variable('act_step',np.ones(u,np.float32))
            m.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(akida.AKD1500()))
            assert len(m.sequences)==len(m.sequences[0].passes)==1;prefixes.append(bytes(m.sequences[0].program));ram,npid=parts(prefixes[-1]);stride=4+n//32
            assert len(ram)==len(packed(w,k))
            assert all(a==b for j,(a,b) in enumerate(zip(ram,packed(w,k))) if j>=2 and (j-2)%stride>=3),'initialized learning thresholds/weights'
            m.map(None)
        fixed(m,'native_prediction_and_history_relay',*mats[3])
        rams=[parts(p)[0] for p in prefixes]
        test=simulate(rams,'gate',arm)['inputs'];v=test.astype('i4')
        v=(v@mats[0][0].astype('i4')>mats[0][1]).astype('i4');v=(v@mats[1][0].astype('i4')>mats[1][1]).astype('i4')
        for k,(n,u,plastic) in enumerate([(N0,U0,P0),(N1,U1,P1)]):
            t=np.full(u,31);t[plastic]=27;v=(v@weights(seed,k).astype('i4')>t).astype('i4')
            w,t=mats[2+k];v=(v@w.astype('i4')>t).astype('i4')
        actual=m.forward(test.reshape(-1,1,1,N)).reshape(-1,N);assert np.array_equal(actual,v),'full virtual nonlearning circuit'
        m.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(akida.AKD1500()))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,banks,pm,adapter=bind(prefixes,bytes(m.sequences[0].program));(out/'model.bin').write_bytes(program)
        layout=['128 2 4']
        for k,b in enumerate(banks):
            for ext in ['ram','infer','learn']:np.asarray(b[ext],'<u4').tofile(out/f'bank{k}.{ext}')
            layout.append(' '.join(map(str,[b['n'],b['u']]+b['np']+[len(b['ram'])])))
        for j,b in enumerate(pm):np.asarray(b['packet'],'<u4').tofile(out/f'fixed{j}.ram')
        (out/'layout').write_text('\n'.join(layout)+'\n')
        for k,(w,t) in enumerate(mats):w.tofile(out/f'fixed{k}.weights');t.tofile(out/f'fixed{k}.threshold')
        cells={}
        for stage in STAGES:
            a=simulate(rams,stage,arm);cells[stage]=assess(a)
            np.asarray([[e['key'],int(e['teacher']),e['label'],int(arm=='severed' and e['teacher'] and e['phase'] not in ['bootstrap','gate'])] for e in a['events']],'<i4').tofile(out/f'{stage}.plan')
            for k,v in a.items():
                if k!='events':v.tofile(out/f'{stage}.{k}')
        r.update(status='VIRTUAL_COMPILER_PASS',mapping=mapping(m),banks=[dict(np=b['np'],n=b['n'],u=b['u'],words=len(b['ram'])) for b in banks],adapter=adapter,fixed_packets=[dict(np=b['np'],words=len(b['packet'])) for b in pm],cells=cells)
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int,choices=SEEDS);p.add_argument('arm',choices=ARMS);a=p.parse_args();main(a.out,a.seed,a.arm)
