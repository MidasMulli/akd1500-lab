"""Compiler/virtual-only spatial fabric fixtures. Never opens a physical device."""
from pathlib import Path
import argparse, hashlib, json, sys, traceback
import numpy as np
import akida
from coprocessor.prepare import Select, conv
from tools.prepare import PARAMS, parts
from tools.program_codec import read, words
from metal.prepare import mapping

P=Path(__file__).resolve().parents[1]

def packets(program, learner):
    header,data=read(program); result=[]
    assert len(header['passes'])==1 and header['learning']['np']==learner
    for record in header['passes'][0]:
        for span in record:
            a=words(data,span);at=0
            while at<len(a):
                h,s=a[at:at+2];length=(s>>16)&0x3fff
                if s>>31:length*=16
                packet=a[at:at+length+2];assert len(packet)==length+2
                mask,uid=(h>>8)&15,h&15
                if mask and uid in [0,2,4,6]:
                    assert mask&(mask-1)==0
                    npid=[h>>24,(h>>16)&255,mask.bit_length()-1]
                    if npid!=learner:result.append((npid,packet))
                at+=len(packet)
            assert at==len(a)
    return result

def pool(z):
    n,h,w,c=z.shape
    return z.reshape(n,h//2,2,w//2,2,c).max(axis=(2,4))

def main(out,side,seed):
    c=512
    assert side in [32,64,96,128]
    out.mkdir(parents=True,exist_ok=False)
    f=out/'cell'; report=dict(filters=c,side=side,seed=seed,physical_device_opened=False)
    try:
        rng=np.random.default_rng(seed); width=32; n=(side//8)**2*8;u=128
        assert u%2==0 and n%32==0
        # Eight recurring cues plus fresh held-out inputs. Queries never teach.
        cues=rng.integers(0,256,(8,side,side,3),np.uint8)
        x=np.concatenate([cues,cues.repeat(4,axis=0),cues,cues,rng.integers(0,256,(8,side,side,3),np.uint8)])
        assert len(x)==64
        labels=np.concatenate([np.arange(8)%2,np.repeat(np.arange(8)%2,4),np.arange(8)%2,np.arange(8)%2,np.arange(8)%2]).astype('<i4')
        w0=rng.integers(-3,4,(3,3,3,width),np.int8)
        t0=np.median(conv(x[:8],w0,None),axis=(0,1,2)).astype('<i4');a0=conv(x,w0,t0)
        w1=rng.integers(-7,8,(3,3,width,c),np.int8)
        z1=conv(a0,w1[::-1,::-1],None);pooled=pool(z1)
        t1=np.median(pooled[:8],axis=(0,1,2)).astype('<i4');a1=(pooled>t1).astype(np.uint8)
        m=akida.Model();m.add(akida.InputConvolutional(name='rgb_events',input_shape=(side,side,3),filters=width,kernel_size=(3,3),kernel_stride=(2,2),weights_bits=8,act_bits=1))
        l=m.get_layer('rgb_events');l.set_variable('weights',w0);l.set_variable('threshold',t0);l.set_variable('act_step',np.ones(width,np.float32))
        assert np.array_equal(m.forward(x),a0),'independent HRC contract'
        m.add(akida.Convolutional(name='resident_spatial_fabric',filters=c,kernel_size=(3,3),kernel_stride=(2,2),weights_bits=4,pool_size=(2,2),pool_stride=(2,2),pool_type=akida.PoolType.Max,act_bits=1))
        l=m.get_layer('resident_spatial_fabric');l.set_variable('weights',w1);l.set_variable('threshold',t1);l.set_variable('act_step',np.ones(c,np.float32))
        assert np.array_equal(m.forward(x),a1),'independent CNP convolution/pool contract'
        # Native max pooling requires a following convolution. This projection
        # combines all spatial channels; it is not a host feature reduction.
        w2=rng.integers(-1,2,(1,1,c,8),np.int8)
        z2=a1.astype(np.int32)@w2[0,0].astype(np.int32)
        t2=np.median(z2[:8],axis=(0,1,2)).astype('<i4');a2=(z2>t2).astype(np.uint8)
        m.add(akida.Convolutional(name='resident_projection',filters=8,kernel_size=(1,1),weights_bits=2,act_bits=1))
        l=m.get_layer('resident_projection');l.set_variable('weights',w2);l.set_variable('threshold',t2);l.set_variable('act_step',np.ones(8,np.float32))
        assert np.array_equal(m.forward(x),a2),'independent native projection contract'
        m.add(akida.FullyConnected(name='resident_native_learner',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS))
        lw=np.zeros((n,u),np.int8)
        for j in range(u):lw[rng.choice(n,32,False),j]=1
        l=m.get_layer('resident_native_learner');l.set_variable('weights',lw.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
        features=a2.transpose(0,3,2,1).reshape(64,n).copy();expected=features.astype(np.int32)@lw.astype(np.int32)
        assert np.array_equal(m.forward(x).reshape(64,u),expected),'full independent virtual contract'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1,'one resident hardware pass'
        program=bytes(m.sequences[0].program);ram,npid=parts(program)
        assert npid==[3,3,0] and len(ram)==2+u*(4+n//32),'FNP3 local RAM contract'
        pm=packets(program,npid)
        assert pm and all(p!=npid for p,_ in pm)
        for ext,data in [('bin',program),('ram',np.asarray(ram,'<u4').tobytes()),('inputs',x.tobytes()),('features',features.tobytes()),('expected',expected.astype('<i4').tobytes()),('labels',labels.tobytes()),('w0',w0.tobytes()),('w1',w1.tobytes()),('w2',w2.tobytes()),('t0',t0.tobytes()),('t1',t1.tobytes()),('t2',t2.tobytes())]:f.with_suffix('.'+ext).write_bytes(data)
        f.with_suffix('.meta').write_text(f'{side} {width} {c} {n} {u} 32 2 64\n')
        f.with_suffix('.layout').write_text(str(len(pm))+'\n'+'\n'.join(' '.join(map(str,p+[len(a)])) for p,a in pm)+'\n')
        for i,(p,a) in enumerate(pm):f.with_suffix(f'.packet{i}.ram').write_bytes(np.asarray(a,'<u4').tobytes())
        report.update(status='VIRTUAL_COMPILER_PASS',mapping=mapping(m),n=n,u=u,learner_np=npid,learner_packet_bytes=len(ram)*4,feature_density=float(features.mean()),fixed_packets=[dict(np=p,bytes=len(a)*4) for p,a in pm],nominal_dense_macs=(side//2)**2*27*width+(side//4)**2*9*width*c+(side//8)**2*c*8)
    except Exception as e:
        report.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    report['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['status']!='VIRTUAL_COMPILER_PASS':sys.exit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('side',type=int,choices=[32,64,96,128]);p.add_argument('seed',type=int);a=p.parse_args();main(a.output,a.side,a.seed)
