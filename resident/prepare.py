"""CNP categorical communication between native FNP3 learning banks."""
from pathlib import Path
import hashlib
import json
import struct
import sys
import traceback
import numpy as np
import akida
from resident.design import N,U,NW,C,T,COUNTS,SEEDS,initial,schedule
from resident.partition import S
from metal.prepare import mapping

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from program_codec import read,build,words
from prepare_network import TARGETS
from prepare_reduction import assemble


class Select(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        if type==akida.NP.Type.FNP3:
            found=[p for target in TARGETS for p in source_nps if [p.col,p.row,p.id]==target]
        else:
            assert type in [akida.NP.Type.CNP1,akida.NP.Type.CNP2],str(type)
            found=[p for p in source_nps if [p.col,p.row,p.id] not in TARGETS]
        assert len(found)>=num_nps,('native resources',str(type),num_nps,len(found))
        return found[:num_nps]


def conv(m,name,w,t):
    m.add(akida.Convolutional(name=name,filters=w.shape[1],kernel_size=(1,1),weights_bits=4,activation=True,act_bits=1))
    l=m.get_layer(name);l.set_variable('weights',w.reshape(1,1,*w.shape).astype(np.int8));l.set_variable('threshold',t.astype(np.int32));l.set_variable('act_step',np.ones(w.shape[1],np.float32))


def communication(m,k):
    pairs=[(i,j) for i in range(C) for j in range(C) if i!=j]
    w=np.zeros((U,32),np.int8);t=np.zeros(32,np.int32)
    for col,(i,j) in enumerate(pairs):
        w[i*16:(i+1)*16,col]=1;w[j*16:(j+1)*16,col]=-1;t[col]=-1 if i<j else 0
    conv(m,f'compare{k}',w,t)
    w=np.zeros((32,N),np.int8)
    for c in range(C):w[c*3:(c+1)*3,c*128:(c+1)*128]=1
    conv(m,f'categorical{k}',w,np.full(N,2,np.int32))


def bind(prefixes,app):
    image,audit=assemble(prefixes[-1],app,N,U);h,data=read(image);data=bytearray(data);banks=[]
    for index,prefix in enumerate(prefixes):
        si,sd=read(prefix);lr=si['learning'];ram=words(sd,lr['ram'][0]);inf=words(sd,lr['inference']);lea=words(sd,lr['learning'])
        if index==len(prefixes)-1:
            ni=words(data,h['learning']['inference']);nl=words(data,h['learning']['learning'])
        else:
            candidates=[(ri,words(data,sp)) for ri,record in enumerate(h['passes'][0]) for sp in record
                if len(words(data,sp))==52+len(ram) and words(data,sp)[0]==inf[0] and words(data,sp)[52:54]==ram[:2]]
            assert len(candidates)==1,('learning record',index,len(candidates));ri,packet=candidates[0];ni=packet[:52];nl=ni[:]
            assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b),'unreviewed route difference'
            assert all(j>=2 and (j-2)%20 in [1,2,3] for j,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b),'unreviewed RAM difference'
            # Preserve compiler-current intermediate output auxiliaries; restore only learning parameters.
            for j in range(U):
                for word in [1]:ram[2+j*20+word]=packet[52+2+j*20+word]
            for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
            span=[len(data),len(ni)+len(ram)];data.extend(struct.pack('<'+'I'*(len(ni)+len(ram)),*(ni+ram)));h['passes'][0][ri]=[span]
        banks.append(dict(np=lr['np'],ram=ram,infer=ni,learn=nl))
    struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));hh,dd=read(image);assert hh==h and dd==bytes(data)
    fixed=[]
    for rec in h['passes'][0]:
        for sp in rec:
            a=words(data,sp);at=0
            while at<len(a):
                header,size=a[at:at+2];length=(size>>16)&0x3fff
                if size>>31:length*=16
                packet=a[at:at+2+length];assert len(packet)==2+length;at+=len(packet)
                mask,uid=(header>>8)&15,header&15
                if mask and uid in [0,2,4,6]:
                    assert not mask&(mask-1),'multicast RAM requires explicit adapter'
                    npid=[header>>24,(header>>16)&255,mask.bit_length()-1]
                    if npid not in [b['np'] for b in banks]:fixed.append(dict(np=npid,packet=packet))
            assert at==len(a)
    return image,banks,fixed,audit


def main():
    out=S/'fixtures-chain03';out.mkdir(exist_ok=False);result=[]
    cpu=json.loads((S/'evidence/CPU_TASK_SCREEN.json').read_text())
    assert all(r['gate']['status']=='PASS' for r in cpu['results'])
    # Map every planned width before seed expansion; fresh weights share topology.
    for count in COUNTS:
        for seed in SEEDS:
            if seed!=SEEDS[0] and count!=6:continue
            name=f'b{count}_s{seed}';m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));prefixes=[];ws=[]
            try:
                for k,ram in enumerate(initial(seed,count)):
                    if k:communication(m,k-1)
                    m.add(akida.FullyConnected(name=f'memory{k}',units=U,weights_bits=1,activation=True,act_bits=1))
                    m.compile(optimizer=akida.AkidaUnsupervised(num_weights=NW,num_classes=C,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
                    l=m.get_layer(f'memory{k}');w=np.zeros((N,U),np.int8)
                    for j in range(U):
                        for bit in range(N):w[bit,j]=(ram[2+j*20+4+bit//32]>>(bit%32))&1
                    l.set_variable('weights',w.reshape(1,1,N,U));l.set_variable('threshold_learning',np.zeros(U,np.int32));l.set_variable('threshold',np.full(U,T,np.int32));l.set_variable('act_step',np.ones(U,np.float32));ws.append(w)
                    d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
                    assert len(m.sequences)==len(m.sequences[0].passes)==1,'single prefix pass';prefixes.append(bytes(m.sequences[0].program))
                rw=np.zeros((U,C),np.int8)
                for c in range(C):rw[c*16:(c+1)*16,c]=1
                m.add(akida.FullyConnected(name='destination_support',units=C,weights_bits=1,activation=False));m.get_layer('destination_support').set_variable('weights',rw.reshape(1,1,U,C))
                x,_=schedule(count,'repair');x=x[:16];v=x.astype(np.int32)
                for w in ws:
                    y=(v@w.astype(np.int32)>T).reshape(-1,C,16).sum(axis=2);v=np.zeros((len(x),N),np.int32)
                    for i,c in enumerate(y.argmax(axis=1)):v[i,c*128:(c+1)*128]=1
                m.map(None);actual=m.forward(x.reshape(-1,1,1,N)).reshape(-1,C);assert np.array_equal(actual,y),'complete virtual chain'
                d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
                assert len(m.sequences)==len(m.sequences[0].passes)==1
                app=bytes(m.sequences[0].program);image,banks,fixed,audit=bind(prefixes,app)
                layout=[f'{N} {U} {C} {count} {len(fixed)} {T}']
                for k,b in enumerate(banks):
                    for ext in ['ram','infer','learn']:(out/f'{name}.bank{k}.{ext}').write_bytes(np.asarray(b[ext],'<u4').tobytes())
                    layout.append(' '.join(map(str,b['np']+[len(b['ram'])])))
                for k,f in enumerate(fixed):
                    (out/f'{name}.fixed{k}.ram').write_bytes(np.asarray(f['packet'],'<u4').tobytes());layout.append(' '.join(map(str,f['np']+[len(f['packet'])])))
                (out/f'{name}.layout').write_text('\n'.join(layout)+'\n');(out/f'{name}.bin').write_bytes(image)
                meta=dict(status='COMPILER_VIRTUAL_ADAPTER_PASS',name=name,seed=seed,banks=count,mapping=mapping(m),learning_nps=[b['np'] for b in banks],
                    fixed_packets=[dict(np=f['np'],words=len(f['packet']),sha256=hashlib.sha256(np.asarray(f['packet'],'<u4').tobytes()).hexdigest()) for f in fixed],
                    bank_ram_sha256=[hashlib.sha256(np.asarray(b['ram'],'<u4').tobytes()).hexdigest() for b in banks],program_sha256=hashlib.sha256(image).hexdigest(),adapter=audit)
                save=out/f'{name}.json';save.write_text(json.dumps(meta,indent=2)+'\n');result.append(meta)
                print(json.dumps(dict(name=name,status=meta['status'],learning_nps=meta['learning_nps'],fixed_regions=len(fixed),physical_nps=len({tuple(p['np']) for r in meta['mapping'] for p in r['nps']}))),flush=True)
            except Exception as e:
                row=dict(name=name,seed=seed,banks=count,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc());result.append(row);print(json.dumps(row),flush=True)
                break
            (out/'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=result),indent=2)+'\n')
        (out/'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=result),indent=2)+'\n')


if __name__=='__main__':main()
