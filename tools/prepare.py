"""Compiler-only capacity frontier. No physical-device discovery or open calls."""
from pathlib import Path
import argparse,hashlib,json,struct,traceback
import akida,numpy as np
P=Path(__file__).resolve().parents[1]
PARAMS=dict(num_weights=32,num_classes=2,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25)
class Table:
    def __init__(self,b,p):self.b=b;self.p=p;self.v=p-struct.unpack_from('<i',b,p)[0]
    def field(self,n):
        at=4+2*n
        if at>=struct.unpack_from('<H',self.b,self.v)[0]:return 0
        k=struct.unpack_from('<H',self.b,self.v+at)[0];return self.p+k if k else 0
    def ptr(self,n):
        p=self.field(n);return p+struct.unpack_from('<I',self.b,p)[0] if p else 0
    def table(self,n):
        p=self.ptr(n);return Table(self.b,p) if p else None
    def tables(self,n):
        p=self.ptr(n)
        if not p:return []
        count=struct.unpack_from('<I',self.b,p)[0];assert count<=128
        return [Table(self.b,p+4+4*i+struct.unpack_from('<I',self.b,p+4+4*i)[0]) for i in range(count)]
    def span(self,n):return struct.unpack_from('<II',self.b,self.field(n))
    def spans(self,n):
        p=self.ptr(n);count=struct.unpack_from('<I',self.b,p)[0]
        return [struct.unpack_from('<II',self.b,p+4+8*i) for i in range(count)]
def parts(image):
    offset=struct.unpack_from('<I',image)[0]+4
    root=Table(image,4+struct.unpack_from('<I',image,4)[0]);learn=root.table(15)
    assert learn is not None
    record=learn.table(2)
    if record.field(1):
        raise NotImplementedError('Compiler selected FNP2 external-memory record; FNP3 decoder does not cover it')
    span=record.spans(0);assert len(span)==1
    a,count=span[0];ram=list(struct.unpack_from('<'+'I'*count,image,offset+a))
    npid=list(image[learn.field(3):learn.field(3)+3])
    return ram,npid
def generate(out):
    out.mkdir(parents=True,exist_ok=False);results=[]
    # Includes a control-sized cell and successively more memory/compute per bank.
    cells=[(128,8),(512,64),(512,256),(512,512),(1024,256),(1024,512),(2048,256),(4096,128),(8192,64),(512,1024),(1024,1024),(2048,1024),(4096,1024),(8192,1024)]
    for n,u in cells:
        name=f'n{n}_u{u}';rng=np.random.default_rng(72301+n+u)
        compiler_passed=False
        try:
            m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1))
            m.add(akida.FullyConnected(name='association',units=u,weights_bits=1,activation=False))
            m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('association')
            w=np.zeros((n,u),np.int8)
            for j in range(u):w[rng.choice(n,32,replace=False),j]=1
            l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
            initial=l.get_variable('weights').reshape(n,u)
            x=np.zeros((1024,n),np.uint8)
            for i in range(1024):x[i,rng.choice(n,max(32,n//4),replace=False)]=1
            scores=m.forward(x[:8].reshape(8,1,1,n)).reshape(8,u)
            assert np.array_equal(scores,x[:8].astype(np.int32)@initial.astype(np.int32))
            m.save(str(out/(name+'.fbz')));m.map(akida.AKD1500(),hw_only=True)
            assert len(m.sequences)==1 and len(m.sequences[0].passes)==1,'not single pass'
            image=bytes(m.sequences[0].program);compiler_passed=True;ram,npid=parts(image)
            assert len(ram)==2+u*(4+n//32),(len(ram),n,u)
            (out/(name+'.bin')).write_bytes(image)
            (out/(name+'.ram')).write_bytes(struct.pack('<'+'I'*len(ram),*ram))
            (out/(name+'.meta')).write_text(f'{n} {u} 32 2\n')
            (out/(name+'.inputs')).write_bytes(x.tobytes())
            (out/(name+'.labels')).write_bytes(np.array([i%2 for i in range(1024)],dtype='<i4').tobytes())
            results.append(dict(name=name,status='MAPPED_SINGLE_PASS',inputs=n,neurons=u,selected_connections=32*u,weight_packet_bytes=4*len(ram),learning_np=npid,program_sha256=hashlib.sha256(image).hexdigest()))
        except Exception as e:
            results.append(dict(name=name,status='MAPPED_UNSUPPORTED_BY_DECODER' if compiler_passed else 'COMPILER_REJECTED',inputs=n,neurons=u,error=str(e),traceback=traceback.format_exc()))
        (out/'CAPACITY.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,parameters=PARAMS,cells=results),indent=2)+'\n')
        print(json.dumps({k:v for k,v in results[-1].items() if k!='traceback'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);generate(p.parse_args().output)
