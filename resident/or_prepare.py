"""Three FNP-only native learning banks; preserve full terminal-prefix auxiliaries."""
import json
import struct
import sys
import numpy as np
import akida
from resident.or_design import P,S,WIDTHS,SEEDS,initial,schedule
from resident.run import save,digest
sys.path.insert(0,str(P/'tools'))
from program_codec import read,build,words
from prepare_network import TARGETS,fc
from prepare_reduction import assemble
from metal.prepare import mapping


class Select(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        assert type==akida.NP.Type.FNP3,str(type)
        found=[p for target in TARGETS for p in source_nps if [p.col,p.row,p.id]==target]
        assert len(found)>=num_nps
        return found[:num_nps]


def bind(prefixes,app,width):
    image,audit=assemble(prefixes[-1],app,512,width);h,data=read(image);data=bytearray(data);banks=[]
    for k,prefix in enumerate(prefixes):
        hh,dd=read(prefix);lr=hh['learning'];ram=words(dd,lr['ram'][0]);inf=words(dd,lr['inference']);lea=words(dd,lr['learning'])
        if k==len(prefixes)-1:ni=words(data,h['learning']['inference']);nl=words(data,h['learning']['learning'])
        else:
            candidates=[(j,words(data,sp)) for j,rec in enumerate(h['passes'][0]) for sp in rec if len(words(data,sp))==52+len(ram) and words(data,sp)[0]==inf[0] and words(data,sp)[52:54]==ram[:2]]
            assert len(candidates)==1;ri,packet=candidates[0];ni=packet[:52];nl=ni[:]
            assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b)
            assert all(j>=2 and (j-2)%20 in [1,2,3] for j,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b)
            for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
            sp=[len(data),52+len(ram)];data.extend(struct.pack('<'+'I'*(52+len(ram)),*(ni+ram)));h['passes'][0][ri]=[sp]
        banks.append(dict(np=lr['np'],ram=ram,infer=ni,learn=nl))
    struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));assert read(image)==(h,bytes(data));fixed=[]
    for rec in h['passes'][0]:
        for span in rec:
            a=words(data,span);at=0
            while at<len(a):
                header,size=a[at:at+2];length=(size>>16)&0x3fff
                if size>>31:length*=16
                packet=a[at:at+2+length];assert len(packet)==2+length;at+=len(packet);mask,uid=(header>>8)&15,header&15
                if mask and uid in [0,2,4,6]:
                    assert not mask&(mask-1);npid=[header>>24,(header>>16)&255,mask.bit_length()-1]
                    if npid not in [b['np'] for b in banks]:fixed.append(dict(np=npid,packet=packet))
            assert at==len(a)
    assert len(fixed)==3 and len({tuple(p['np']) for p in fixed})==3
    return image,banks,fixed,audit


def main():
    out=S/'fixtures01';out.mkdir(exist_ok=False);cpu=json.loads((S/'evidence/CPU_SCREEN.json').read_text());assert all(r['gate']['status']=='PASS' for r in cpu['results']);results=[]
    for width in WIDTHS:
        for seed in SEEDS:
            if width!=512 and seed!=SEEDS[0]:continue
            name=f'u{width}_s{seed}';m=akida.Model();m.add(akida.InputData(input_shape=(1,1,512),input_bits=1));prefixes=[];weights=[]
            for k,ram in enumerate(initial(seed,width)):
                if k:
                    w=np.zeros((width,512),np.int8)
                    for c in range(4):w[c*(width//4):(c+1)*(width//4),c*128:(c+1)*128]=1
                    fc(m,f'presence{k-1}',w,np.zeros(512,np.int32),bits=1)
                m.add(akida.FullyConnected(name=f'memory{k}',units=width,weights_bits=1,activation=True,act_bits=1));m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=4,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
                l=m.get_layer(f'memory{k}');w=np.zeros((512,width),np.int8)
                for j in range(width):
                    for bit in range(512):w[bit,j]=(ram[2+j*20+4+bit//32]>>(bit%32))&1
                l.set_variable('weights',w.reshape(1,1,512,width));l.set_variable('threshold_learning',np.zeros(width,np.int32));l.set_variable('threshold',np.full(width,27,np.int32));l.set_variable('act_step',np.ones(width,np.float32));weights.append(w)
                d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1;prefixes.append(bytes(m.sequences[0].program))
            w=np.zeros((width,4),np.int8)
            for c in range(4):w[c*(width//4):(c+1)*(width//4),c]=1
            fc(m,'readout',w,None,activation=False,bits=1)
            x,_=schedule('repair');test=np.vstack([x[:16],weights[0][:,[0,width//4,width//2,3*width//4]].T]).astype(np.uint8);v=test.astype(np.int32)
            for ww in weights:
                support=(v@ww.astype(np.int32)>27).reshape(-1,4,width//4).sum(2);v=np.repeat(support>0,128,axis=1).astype(np.int32)
            m.map(None);actual=m.forward(test.reshape(-1,1,1,512)).reshape(-1,4);assert np.array_equal(actual,support),'virtual FNP-only graph'
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
            image,banks,fixed,audit=bind(prefixes,bytes(m.sequences[0].program),width);layout=[f'512 {width} 4 3 3 27']
            for j,b in enumerate(banks):
                for ext in ['ram','infer','learn']:(out/f'{name}.bank{j}.{ext}').write_bytes(np.asarray(b[ext],'<u4').tobytes())
                layout.append(' '.join(map(str,b['np']+[len(b['ram'])])))
            for j,b in enumerate(fixed):
                (out/f'{name}.fixed{j}.ram').write_bytes(np.asarray(b['packet'],'<u4').tobytes());layout.append(' '.join(map(str,b['np']+[len(b['packet'])])))
            f=out/name;f.with_suffix('.layout').write_text('\n'.join(layout)+'\n');f.with_suffix('.bin').write_bytes(image)
            r=dict(name=name,width=width,seed=seed,status='COMPILER_VIRTUAL_ADAPTER_PASS',mapping=mapping(m),learning_nps=[b['np'] for b in banks],
                bank_ram_sha256=[digest(out/f'{name}.bank{j}.ram') for j in range(3)],fixed_packets=[dict(np=b['np'],sha256=digest(out/f'{name}.fixed{j}.ram')) for j,b in enumerate(fixed)],adapter=audit,program_sha256=digest(f.with_suffix('.bin')))
            save(f.with_suffix('.json'),r);results.append(r);save(out/'BUILD.json',dict(status='COMPILER_ONLY',physical_device_opened=False,results=results));print(json.dumps(dict(name=name,status=r['status'],learning_nps=r['learning_nps'])),flush=True)


if __name__=='__main__':main()
