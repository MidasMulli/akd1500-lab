"""Compiler-only intermediate learner + fixed two-count output reducer."""
from pathlib import Path
import argparse,copy,hashlib,json,struct,traceback
import akida,numpy as np
from program_codec import read,build,words
from prepare import parts

def assemble(single,app,n,u):
    si,sd=read(single);h,data=read(app);lr=si['learning']
    assert lr and h['learning'] is None and len(h['passes'])==1
    inf=words(sd,lr['inference']);lea=words(sd,lr['learning']);ram=words(sd,lr['ram'][0])
    assert len(inf)==len(lea)==52 and len(ram)==2+u*(4+n//32)
    modes=[i for i,(a,b) in enumerate(zip(inf,lea)) if a!=b];assert modes==[38,39,40],modes
    candidates=[]
    for ri,rec in enumerate(h['passes'][0]):
        for sp in rec:
            t=words(data,sp)
            if len(t)==52+len(ram) and t[0]==inf[0] and t[52:54]==ram[:2]:candidates.append((ri,t))
    assert len(candidates)==1,[(ri,len(t)) for ri,t in candidates]
    ri,t=candidates[0];routes=[i for i,(a,b) in enumerate(zip(inf,t[:52])) if a!=b]
    assert all(i in [46,47,49] for i in routes),routes
    stride=4+n//32;diffs=[i for i,(a,b) in enumerate(zip(ram,t[52:])) if a!=b]
    assert all(i>=2 and (i-2)%stride in [1,2,3] for i in diffs),diffs[:30]
    ni=t[:52];nl=ni[:]
    for k in modes:assert ni[k]==inf[k];nl[k]=lea[k]
    d=bytearray(data)
    def add(v):
        sp=[len(d),len(v)];d.extend(struct.pack('<'+'I'*len(v),*v));return sp
    ls=add(nl);ins=add(ni);rs=add(ram);struct.pack_into('<I',d,0,len(d)-4)
    a=copy.deepcopy(h);a['learning']=dict(learning=ls,inference=ins,ram=[rs],np=lr['np'])
    result=build(a,bytes(d));aa,dd=read(result)
    assert aa==a and aa['passes']==h['passes']
    assert all(words(dd,s)==words(data,s) for ps in h['passes'] for rec in ps for s in rec)
    return result,dict(status='PASS',learning_np=lr['np'],mode_word_indices=modes,route_word_indices=routes,restored_auxiliary_words=len(diffs),inference_tracks_unchanged=True)

def main(source,out):
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for n,u in [(128,8),(512,512)]:
        name=f'n{n}_u{u}_reduced';f=source/f'n{n}_u{u}_mask'
        try:
            m=akida.Model(str(f.with_suffix('.fbz')))
            rw=np.zeros((u,2),np.int8);rw[:u//2,0]=1;rw[u//2:,1]=1
            m.add(akida.FullyConnected(name='fixed_class_support',units=2,weights_bits=1,activation=False))
            l=m.get_layer('fixed_class_support');l.set_variable('weights',rw.reshape(1,1,u,2))
            x=np.frombuffer(f.with_suffix('.inputs').read_bytes(),np.uint8).reshape(-1,n)
            initial=m.get_layer('association').get_variable('weights').reshape(n,u).astype(np.int32)
            expected=(x[:16].astype(np.int32)@initial>7).astype(np.int32)@rw.astype(np.int32)
            got=m.forward(x[:16].reshape(-1,1,1,n)).reshape(-1,2);assert np.array_equal(got,expected),'virtual reduction'
            m.save(str(out/(name+'.fbz')));m.map(akida.AKD1500(),hw_only=True)
            assert len(m.sequences)==1 and len(m.sequences[0].passes)==1
            mapping=[]
            for layer in m.sequences[0].passes[0].layers:
                components=[]
                for p in layer.mapping.nps:
                    q=p.ident;components.append(dict(type=str(p.type),np=[q.col,q.row,q.id]))
                mapping.append(dict(layer=layer.name,components=components))
            app=bytes(m.sequences[0].program);(out/(name+'.inference.bin')).write_bytes(app)
            image,audit=assemble(f.with_suffix('.bin').read_bytes(),app,n,u)
            ram,npid=parts(image);assert ram==list(struct.unpack('<'+'I'*(len(f.with_suffix('.ram').read_bytes())//4),f.with_suffix('.ram').read_bytes()))
            (out/(name+'.bin')).write_bytes(image)
            for ext in ['ram','inputs','labels','meta']:(out/(name+'.'+ext)).write_bytes(f.with_suffix('.'+ext).read_bytes())
            (out/(name+'.output')).write_text('2 7\n')
            row=dict(name=name,status='MAPPED',inputs=n,neurons=u,outputs=2,mapping=mapping,audit=audit,program_sha256=hashlib.sha256(image).hexdigest())
        except Exception as e:
            row=dict(name=name,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
        rows.append(row);(out/'REDUCTION.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=rows),indent=2)+'\n');print(json.dumps(row),flush=True)
    assert all(r['status']=='MAPPED' for r in rows),'compiler gate failed; no hardware admission'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
