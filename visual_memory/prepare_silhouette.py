from pathlib import Path
import argparse,json,struct,sys,traceback
import numpy as np
import akida
from visual_memory.silhouette import *
from visual_memory.silhouette_study import rows
from visual_memory.data import save,sha
from visual_memory.prepare_v4 import fixed_packets
from tools.program_codec import read,build,words
from tools.prepare_network import TARGETS
from metal.prepare import mapping
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tools.prepare_reduction import assemble
class Select(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        if type==akida.NP.Type.FNP3:found=[q for target in TARGETS for q in source_nps if [q.col,q.row,q.id]==target]
        elif type==akida.NP.Type.FNP2:found=[q for q in source_nps if [q.col,q.row,q.id]==[1,2,0]]
        else:found=list(source_nps)
        assert len(found)>=num_nps,(str(type),num_nps,len(found));return found[:num_nps]
def mapped(m):
    d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1;return bytes(m.sequences[0].program)
def plastic(m,k):
    n,u,nw,classes=dimensions(k)
    m.add(akida.FullyConnected(name=f'plastic_{k}',units=u,weights_bits=1,activation=True,act_bits=4 if k==0 else 1))
    m.compile(optimizer=akida.AkidaUnsupervised(num_weights=nw,num_classes=classes,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
    l=m.layers[-1];l.set_variable('weights',weights(k).reshape(1,1,n,u));l.set_variable('threshold_learning',learning_thresholds(k));l.set_variable('threshold',thresholds(k));l.set_variable('act_step',steps(k))
def add_fixed(m,entry):
    name,w,t,bits=entry;m.add(akida.FullyConnected(name=name,units=w.shape[1],weights_bits=bits,activation=True,act_bits=1));l=m.layers[-1];l.set_variable('weights',w.reshape(1,1,*w.shape));l.set_variable('threshold',t);l.set_variable('act_step',np.ones(w.shape[1],'f4'))
def bind(prefixes,final):
    image,audit=assemble(prefixes[1],final,512,400);h,data=read(image);data=bytearray(data);banks=[]
    for k,prefix in enumerate(prefixes):
        hh,dd=read(prefix);lr=hh['learning'];ram=words(dd,lr['ram'][0]);inf=words(dd,lr['inference']);lea=words(dd,lr['learning'])
        if k==1:ni=words(data,h['learning']['inference']);nl=words(data,h['learning']['learning'])
        else:
            found=[(i,words(data,sp)) for i,rec in enumerate(h['passes'][0]) for sp in rec if len(words(data,sp))==52+len(ram) and words(data,sp)[0]==inf[0] and words(data,sp)[52:54]==ram[:2]];assert len(found)==1
            i,packet=found[0];ni=packet[:52];nl=ni[:];assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b);assert all(j>=2 and (j-2)%68 in [1,2,3] for j,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b)
            for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
            span=[len(data),52+len(ram)];data.extend(np.asarray(ni+ram,'<u4').tobytes());h['passes'][0][i]=[span]
        banks.append(dict(np=lr['np'],ram=ram,infer=ni,learn=nl))
    struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));return image,banks,audit
def main(out):
    out.mkdir(exist_ok=False);r=dict(physical_device_opened=False,record_contents_in_fixed_weights=False,wire='native512rawint32bits=2048binarybytes')
    try:
        entries=list(matrices())
        x=np.stack([encode(p,query=True) for p in patterns()]+[np.zeros(2048,'u1')]).reshape(-1,1,1,2048);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,2048),input_bits=1));v=x.reshape(-1,2048);prefixes=[]
        for k in range(2):
            plastic(m,k);v=activate(v.astype('i4')@weights(k).astype('i4'),k);np.testing.assert_array_equal(m.forward(x).reshape(len(x),-1),v);prefix=mapped(m);prefixes.append(prefix);(out/f'prefix{k}.bin').write_bytes(prefix);m.map(None)
            for entry in entries[:2] if k==0 else entries[2:]:add_fixed(m,entry);v=np.stack([fixed(q,entry) for q in v]);np.testing.assert_array_equal(m.forward(x).reshape(len(x),-1),v)
        m.add(akida.FullyConnected(name='raw_word_feedback_wire',units=512,weights_bits=1,activation=False));m.layers[-1].set_variable('weights',formatter().reshape(1,1,512,512));v=v.astype('i4')@formatter().astype('i4');np.testing.assert_array_equal(m.forward(x).reshape(len(x),512),v)
        c=Circuit()
        for row in rows():
            if row['mode']>=0:c.step(row['input'],row['mode'],row['label'])
        for k,mem in enumerate(c.banks):
            n=dimensions(k)[0];a=np.array([[(int(mask)>>j)&1 for mask in mem.weights] for j in range(n)],'i1');m.get_layer(f'plastic_{k}').set_variable('weights',a.reshape(1,1,*a.shape))
        expected=np.stack([c.step(q.reshape(2048))['output'] for q in x]);np.testing.assert_array_equal(m.forward(x).reshape(len(x),512),expected)
        assert expected[:4,472].all()
        for k in range(2):m.get_layer(f'plastic_{k}').set_variable('weights',weights(k).reshape(1,1,*weights(k).shape))
        compiler=mapped(m);r['mapping']=mapping(m);(out/'compiler.bin').write_bytes(compiler);image,banks,audit=bind(prefixes,compiler);(out/'model.bin').write_bytes(image)
        for k,b in enumerate(banks):
            for ext in ['ram','infer','learn']:np.asarray(b[ext],'<u4').tofile(out/f'bank{k}.{ext}')
            thresholds(k).tofile(out/f'bank{k}.threshold');steps(k).astype('<i4').tofile(out/f'bank{k}.step')
        for k,entry in enumerate(entries+[('raw_formatter',formatter(),np.zeros(512,'i4'),1)]):entry[1].tofile(out/f'fixed_layer{k}.weights');entry[2].tofile(out/f'fixed_layer{k}.threshold')
        packets=[p for p in fixed_packets(image) if p[0] not in [b['np'] for b in banks]]
        for k,p in enumerate(packets):np.asarray(p[1],'<u4').tofile(out/f'fixed{k}.ram')
        (out/'fixed_count.txt').write_text(str(len(packets))+'\n');nps={tuple(q['np']) for l in r['mapping'] for q in l['nps'] if q['type']!='Type.HRC'};r.update(status='SILHOUETTE_VIRTUAL_PASS',single_pass=True,SDK_integer_exact=True,learned_witness_exact=True,initial_weights_restored=True,resident_NPUs=len(nps),fixed_packets=len(packets),binding=audit,program_sha256=sha(out/'model.bin'))
    except BaseException as e:r.update(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc())
    save(out/'BUILD.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['mapping','traceback']},indent=2))
    if r['status']!='SILHOUETTE_VIRTUAL_PASS':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();main(a.out)
