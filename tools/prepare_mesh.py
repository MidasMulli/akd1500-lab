"""Single-pass mesh routing stress, NOT a useful CNN accuracy/speed benchmark.

Each fixed 1x1 layer permutes channels. A final native FNP3 learns binary
associations. The CPU reference composes the fixed permutations algebraically.
"""
from pathlib import Path
import argparse,json,struct,traceback
import akida,numpy as np
from prepare import PARAMS,parts,Table
def main(out):
    out.mkdir(parents=True,exist_ok=False);results=[];n=128;u=64
    reserved=(3,3,0)
    for layers in [1,7,15,31]:
        name=f'mesh_{layers+1}';rng=np.random.default_rng(72400+layers);used=[];calls=[];maps=[]
        try:
            m=akida.Model();m.add(akida.InputData(input_shape=(1,1,128),input_bits=1));perm=np.arange(128)
            for k in range(layers):
                lname=f'route_{k:02}';m.add(akida.Convolutional(name=lname,filters=128,kernel_size=(1,1),weights_bits=1,activation=True,act_bits=1))
                l=m.get_layer(lname);p=rng.permutation(128);w=np.zeros((1,1,128,128),np.int8)
                for j in range(128):w[0,0,p[j],j]=1
                l.set_variable('weights',w);l.set_variable('threshold',np.zeros(128,np.int32));l.set_variable('act_step',np.ones(128,np.float32));perm=perm[p]
            # This runtime's FullyConnected layer flattens its spatial input.
            m.add(akida.FullyConnected(name='memory',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('memory')
            w=np.zeros((n,u),np.int8)
            for j in range(u):w[rng.choice(n,32,replace=False),j]=1
            l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
            x=np.zeros((1024,1,1,128),np.uint8)
            for i in range(1024):x[i].reshape(-1)[rng.choice(n,32,replace=False)]=1
            ref=x[:,:,: ,perm].reshape(1024,n).copy()
            got=m.forward(x[:8]).reshape(8,u);assert np.array_equal(got,ref[:8].astype(np.int32)@w.astype(np.int32))
            class Select(akida.MapConstraints):
                def select_nps(self,source_nps,num_nps,type):
                    # Mapper search can revisit a candidate: source_nps is the
                    # current allocation pool, not a committed-placement callback.
                    candidates=list(source_nps)
                    if type==akida.NP.Type.FNP3:
                        candidates=[q for q in candidates if (q.col,q.row,q.id)==reserved]
                    else:candidates=[q for q in candidates if (q.col,q.row,q.id)!=reserved]
                    chosen=candidates[:num_nps]
                    calls.append(dict(type=str(type),requested=num_nps,candidates=len(candidates),chosen=[[q.col,q.row,q.id] for q in chosen]))
                    if len(chosen)!=num_nps:raise RuntimeError('No unused compatible NP for next layer')
                    used.extend((q.col,q.row,q.id) for q in chosen);return chosen
            device=akida.AKD1500();m.map(device,hw_only=True,constraints=Select(device))
            assert len(m.sequences)==1 and len(m.sequences[0].passes)==1,'multi-pass not admitted'
            b=bytes(m.sequences[0].program);ram,npid=parts(b);assert npid==list(reserved)
            off=struct.unpack_from('<I',b)[0]+4;root=Table(b,4+struct.unpack_from('<I',b,4)[0]);actual=set()
            for ps in root.tables(14):
                for rec in ps.tables(0):
                    assert not rec.field(1),'FNP2 not admitted in mesh integrity test'
                    for at,count in rec.spans(0):
                        words=struct.unpack_from('<'+'I'*count,b,off+at);k=0
                        while k<count:
                            h,sz=words[k:k+2];mask=(h>>8)&15
                            if mask:
                                for j in range(4):
                                    if mask&(1<<j):actual.add((h>>24,(h>>16)&255,j))
                            length=(sz>>16)&0x3fff
                            if sz>>31:length*=16
                            k+=2+length
                        assert k==count,'unparsed configuration payload'
            assignments=[];compute=set()
            for mapped_layer in m.sequences[0].passes[0].layers:
                row=[]
                for component in mapped_layer.mapping.nps:
                    q=component.ident;ident=(q.col,q.row,q.id);compute.add(ident)
                    row.append(dict(np=ident,type=str(component.type),filters=component.filters))
                assignments.append(dict(layer=mapped_layer.name,components=row))
            used=sorted(compute);assert 1<len(used)<=32
            for suffix,data in [('bin',b),('ram',struct.pack('<'+'I'*len(ram),*ram)),('inputs',x.tobytes()),('cpu_inputs',ref.tobytes()),('labels',np.array([i%2 for i in range(1024)],dtype='<i4').tobytes())]:(out/(name+'.'+suffix)).write_bytes(data)
            (out/(name+'.meta')).write_text(f'{n} {u} 32 2\n')
            results.append(dict(name=name,status='MAPPED_SINGLE_PASS',np_count=len(used),nps=used,layer_assignments=assignments,configuration_targets=sorted(actual),learning_np=npid,permutation=perm.tolist(),cpu_equivalence='Fixed permutations composed; synthetic routing integrity, no full CNN performance claim',calls=calls))
        except Exception as e:results.append(dict(name=name,status='NOT_ADMITTED',error=repr(e),traceback=traceback.format_exc(),calls=calls))
        (out/'MESH.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=results),indent=2)+'\n');print(json.dumps({k:v for k,v in results[-1].items() if k not in ['traceback','calls','permutation']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
