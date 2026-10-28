"""Compiler-only own RGB encoder plus present/absent native memory."""
from pathlib import Path
import json
import numpy as np
import akida
from visual_memory.topology_v3 import model
from visual_memory.encoder_v3 import forward
from visual_memory.data import save,sha
from visual_memory.retention_v4 import weights,simulate,native_quality
from coprocessor.prepare import Select
from tools.prepare import parts
from tools.program_codec import read,words
from metal.prepare import mapping
P=Path(__file__).resolve().parents[1];S=P/'studies/20261026-visual-memory-v4'

def fixed_packets(program,learner=None):
    header,data=read(program);out=[];assert len(header['passes'])==1
    for record in header['passes'][0]:
        for span in record:
            a=words(data,span);at=0
            while at<len(a):
                h,s=a[at:at+2];length=(s>>16)&0x3fff
                if s>>31:length*=16
                packet=a[at:at+length+2];assert len(packet)==length+2;mask,uid=(h>>8)&15,h&15
                if mask and uid in [0,2,4,6]:
                    assert mask&(mask-1)==0;npid=[h>>24,(h>>16)&255,mask.bit_length()-1]
                    if npid!=learner:out.append((npid,packet))
                at+=len(packet)
    return out

def packet_files(d,b,learner=None):
    pp=fixed_packets(b,learner);assert pp
    (d/'cell.layout').write_text(str(len(pp))+'\n'+'\n'.join(' '.join(map(str,p+[len(a)])) for p,a in pp)+'\n')
    for j,(p,a) in enumerate(pp):(d/f'cell.packet{j}.ram').write_bytes(np.array(a,'<u4').tobytes())

def prepare():
    screen=S/'development-screen';review=json.loads((screen/'RESULT.json').read_text());assert review['status']=='CPU_RETENTION_GATE_PASS';dual=review['chosen']=='complementary';n=1024 if dual else 512;w=np.load(S/'encoder01/encoder.npz');raw=np.fromfile(screen/'raw-images.u8','u1').reshape(-1,128,128,3);x=raw[:4];f=forward(x,w);expected=np.concatenate([f,1-f],axis=1) if dual else f
    base=model(w,x);e=akida.Model(layers=base.layers[:-1]);assert np.array_equal(e.forward(x).reshape(4,512),f)
    if dual:
        a=np.concatenate([np.eye(512,dtype='i1'),-np.eye(512,dtype='i1')],axis=1).reshape(1,1,512,1024);e.add(akida.Convolutional(name='native_presence_absence',filters=1024,kernel_size=(1,1),weights_bits=2,act_bits=1));l=e.get_layer('native_presence_absence');l.set_variable('weights',a);l.set_variable('threshold',np.r_[np.zeros(512,'i4'),-np.ones(512,'i4')]);l.set_variable('act_step',np.ones(1024,'f4'))
    assert np.array_equal(e.forward(x).reshape(4,n),expected);g=akida.Model(layers=e.layers)
    out=S/'native-fixtures';out.mkdir(exist_ok=False);d=out/'encoder-smoke';d.mkdir();e.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal);assert len(e.sequences)==len(e.sequences[0].passes)==1;b=bytes(e.sequences[0].program);(d/'cell.bin').write_bytes(b);(d/'cell.inputs').write_bytes(x.tobytes());(d/'cell.cpu_inputs').write_bytes(expected.tobytes());(d/'cell.meta').write_text(f'49152 {n} {n} 64 11\n');packet_files(d,b)
    g.add(akida.FullyConnected(name='view_memory',units=220,weights_bits=1,activation=False));g.compile(optimizer=akida.AkidaUnsupervised(num_weights=64,num_classes=11,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=0));l=g.get_layer('view_memory');l.set_variable('weights',weights(dual).reshape(1,1,n,220));l.set_variable('threshold_learning',np.r_[np.zeros(200,'i4'),np.full(20,64,'i4')]);assert np.array_equal(g.forward(x).reshape(4,220),expected.astype('i4')@weights(dual).astype('i4'))
    g.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(akida.AKD1500()))
    assert len(g.sequences)==len(g.sequences[0].passes)==1;b=bytes(g.sequences[0].program);ram,npid=parts(b);assert npid==[3,3,0] and len(ram)==2+220*(4+n//32);ram=np.array(ram,'<u4');assert len(ram)*4<=32768
    rows=json.loads((screen/'ROWS.json').read_text());bits=np.fromfile(screen/review['chosen']/'inputs.u8','u1').reshape(-1,n)
    for tag in ['adaptive','frozen','learning-smoke']:
        d=out/('development-'+tag if tag!='learning-smoke' else tag);d.mkdir();old=np.load(screen/review['chosen']/('adaptive.npz' if tag=='learning-smoke' else tag+'.npz'));ii=np.r_[np.arange(8),np.arange(80,84),np.arange(260,264)] if tag=='learning-smoke' else np.arange(520);xraw=raw[ii];xx=bits[ii];yy=old['labels'][ii];sc,states=simulate(xx,yy,ram)
        if tag!='learning-smoke':assert native_quality(sc,rows)==review['encodings'][review['chosen']]['arms'][tag]
        (d/'cell.bin').write_bytes(b);(d/'cell.inputs').write_bytes(xraw.tobytes());(d/'cell.cpu_inputs').write_bytes(xx.tobytes());(d/'cell.labels').write_bytes(yy.tobytes());(d/'cell.ram').write_bytes(ram.tobytes());(d/'cell.meta').write_text(f'49152 {n} 220 64 11\n');np.savez_compressed(d/'REFERENCE.npz',scores=sc,ram=states);save(d/'ROWS.json',[rows[int(i)] for i in ii]);packet_files(d,b,npid)
    r=dict(status='TRAINED_RGB_VIRTUAL_MAP_PASS',physical_sessions=0,selected=review['chosen'],mapping=mapping(g),learner_np=npid,learner_RAM_bytes=len(ram)*4,all_virtual_layers_exact=True,complement_native=True,single_resident_pass=True,program_bytes=len(b));save(out/'BUILD.json',r);print(json.dumps({k:v for k,v in r.items() if k!='mapping'}),flush=True)

if __name__=='__main__':prepare()
