"""Compiler-routed 2/3-bank FNP3-only native learning network; no physical device open."""
from pathlib import Path
import argparse,copy,hashlib,json,struct,traceback
import akida,numpy as np
from prepare import PARAMS
from prepare_reduction import assemble
from program_codec import read,build,words
TARGETS=[[1,3,0],[2,1,0],[2,2,0],[2,3,0],[3,1,0],[3,2,0],[3,3,0]]
FIXED=[[1,2,1],[1,2,2],[1,2,3],[1,3,1],[1,3,2],[1,3,3],[2,1,1],[2,1,2]]

class Select(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        targets=TARGETS if type==akida.NP.Type.FNP3 else FIXED
        found=[q for target in targets for q in source_nps if [q.col,q.row,q.id]==target]
        assert len(found)>=num_nps,(str(type),num_nps,len(found))
        return found[:num_nps]

def fc(m,name,w,threshold,activation=True,bits=4):
    m.add(akida.FullyConnected(name=name,units=w.shape[1],weights_bits=bits,activation=activation,act_bits=1));l=m.get_layer(name)
    l.set_variable('weights',w.reshape(1,1,*w.shape).astype(np.int8))
    if activation:l.set_variable('threshold',np.asarray(threshold,np.int32));l.set_variable('act_step',np.ones(w.shape[1],np.float32))

def main(out,balanced=False):
    out.mkdir(parents=True,exist_ok=False);rows=[];n=512;u=64
    for seed in [96101,96102,96103]:
        for count in [2,3]:
            name=f'seed{seed}_banks{count}';rng=np.random.default_rng(seed);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));prefixes=[];initial_weights=[]
            try:
                for k in range(count):
                    if k:
                        w=np.zeros((u,n),np.int8);t=np.zeros(n,np.int32)
                        for j in range(u):
                            if balanced:w[j,j*8:j*8+4]=1;w[j,j*8+4:j*8+8]=-1;t[j*8+4:j*8+8]=-1
                            else:w[j,j*8:j*8+8]=1
                        fc(m,f'encode_{k-1}',w,t)
                    m.add(akida.FullyConnected(name=f'memory_{k}',units=u,weights_bits=1,activation=True,act_bits=1));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer(f'memory_{k}')
                    w=np.zeros((n,u),np.int8)
                    for j in range(u):w[rng.choice(n,32,replace=False),j]=1
                    l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32));l.set_variable('threshold',np.full(u,15,np.int32));l.set_variable('act_step',np.ones(u,np.float32));initial_weights.append(w.copy())
                    d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));prefixes.append(bytes(m.sequences[0].program))
                rw=np.zeros((u,2),np.int8);rw[:u//2,0]=1;rw[u//2:,1]=1;fc(m,'readout',rw,None,activation=False)
                x=np.zeros((256,n),np.uint8)
                for i in range(256):x[i,rng.choice(n,n//2,replace=False)]=1
                v=x[:16].astype(np.int32)
                for k,w in enumerate(initial_weights):
                    v=(v@w.astype(np.int32)>15).astype(np.int32)
                    if k<count-1:v=np.stack([v,1-v],axis=2).repeat(4,axis=2).reshape(16,n) if balanced else v.repeat(8,axis=1)
                expected=v@rw.astype(np.int32);m.map(None);got=m.forward(x[:16].reshape(16,1,1,n)).reshape(16,2);assert np.array_equal(got,expected),'virtual integer cascade'
                d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==1 and len(m.sequences[0].passes)==1
                app=bytes(m.sequences[0].program);(out/(name+'.compiler.bin')).write_bytes(app);image,last_audit=assemble(prefixes[-1],app,n,u);h,data=read(image);data=bytearray(data);banks=[]
                for k,prefix in enumerate(prefixes):
                    si,sd=read(prefix);lr=si['learning'];ram=words(sd,lr['ram'][0]);inf=words(sd,lr['inference']);lea=words(sd,lr['learning']);assert lr['np']==TARGETS[2*k]
                    if k==count-1:
                        lr2=h['learning'];ni=words(data,lr2['inference']);nl=words(data,lr2['learning'])
                    else:
                        candidates=[]
                        for ri,rec in enumerate(h['passes'][0]):
                            for sp in rec:
                                t=words(data,sp)
                                if len(t)==52+len(ram) and t[0]==inf[0] and t[52:54]==ram[:2]:candidates.append((ri,t))
                        assert len(candidates)==1,(k,'memory record',len(candidates));ri,t=candidates[0];ni=t[:52];nl=ni[:]
                        assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b)
                        assert all(j>=2 and (j-2)%20 in [1,2,3] for j,(a,b) in enumerate(zip(ram,t[52:])) if a!=b)
                        for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
                        span=[len(data),len(ni)+len(ram)];data.extend(struct.pack('<'+'I'*(len(ni)+len(ram)),*(ni+ram)));h['passes'][0][ri]=[span]
                    for ext,values in [('ram',ram),('infer',ni),('learn',nl)]:(out/f'{name}.bank{k}.{ext}').write_bytes(struct.pack('<'+'I'*len(values),*values))
                    banks.append(dict(index=k,np=lr['np'],ram_words=len(ram)))
                struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));hh,dd=read(image);assert hh==h and dd==bytes(data)
                fixed=[]
                for rec in h['passes'][0]:
                    for sp in rec:
                        t=words(data,sp);at=0
                        while at<len(t):
                            hdr,size=t[at:at+2];length=(size>>16)&0x3fff
                            if size>>31:length*=16
                            packet=t[at:at+2+length];assert len(packet)==2+length;at+=len(packet)
                            mask=(hdr>>8)&15;uid=hdr&15
                            if mask and uid in [0,2,4,6]:
                                assert not(mask&(mask-1));npid=[hdr>>24,(hdr>>16)&255,mask.bit_length()-1]
                                if npid not in [b['np'] for b in banks]:fixed.append(dict(np=npid,header=packet[:2],words=packet[2:]))
                        assert at==len(t)
                assert len({tuple(f['np']) for f in fixed})==count,(len(fixed),count)
                for j,fixed_packet in enumerate(fixed):
                    z=fixed_packet['header']+fixed_packet['words'];(out/f'{name}.fixed{j}.ram').write_bytes(struct.pack('<'+'I'*len(z),*z))
                layout=[f'{n} {u} 2 {count} {len(fixed)}']+[' '.join(map(str,b['np']+[b['ram_words']])) for b in banks]+[' '.join(map(str,f['np']+[len(f['words'])])) for f in fixed]
                (out/(name+'.layout')).write_text('\n'.join(layout)+f'\n{int(balanced)}\n')
                (out/(name+'.bin')).write_bytes(image);(out/(name+'.inputs')).write_bytes(x.tobytes());(out/(name+'.labels')).write_bytes(np.array([i%2 for i in range(256)],dtype='<i4').tobytes())
                plan=dict(encoder="dual_rail" if balanced else "positive_8_copy",name=name,seed=seed,inputs=n,neurons=u,banks=banks,fixed=fixed,threshold=15,classes=2,weights=32,output_contract='two group counts',initial_virtual_outputs=expected.tolist(),configuration=last_audit)
                (out/(name+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
                row=dict(name=name,status='MAPPED_AND_VIRTUAL_PASS',banks=count,configured_learning_neurons=count*u,learning_placements=[b['np'] for b in banks],fixed_placements=sorted({tuple(r['np']) for r in fixed}),program_sha256=hashlib.sha256(image).hexdigest())
            except Exception as e:row=dict(name=name,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
            rows.append(row);(out/'NETWORK.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=rows),indent=2)+'\n');print(json.dumps(row),flush=True)
    assert all(r['status']=='MAPPED_AND_VIRTUAL_PASS' for r in rows),'compiler gate failed; inspect before any hardware'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--balanced',action='store_true');a=p.parse_args();main(a.output,a.balanced)
