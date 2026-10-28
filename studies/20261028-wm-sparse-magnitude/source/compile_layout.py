"""Compiler-only sparse-magnitude geometry qualification. No device open."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from core import Layout

def prepare(out,profile):
    import akida
    from tools.prepare_reduction import assemble
    from visual_memory.separation import save_roles,TARGETS
    from visual_memory.prepare_v4 import fixed_packets
    from tools.program_codec import read,words
    out.mkdir(exist_ok=False);g=Layout(profile);ws,ts,lts=g.parameters();images=[];banks=[];dims=[(g.n0,g.u0),(g.n1,8)];mapping=[];fixed_used={0:set(),1:set()}
    def mapped(m,targets,role,prefix=False):
        class Select(akida.MapConstraints):
            def select_nps(self,source_nps,num_nps,type):
                if num_nps==0:return []
                pool=list(source_nps)
                if type==akida.NP.Type.FNP3:pool=[x for t in targets for x in pool if[x.col,x.row,x.id]==t]
                elif type==akida.NP.Type.FNP2:
                    assert not prefix,'Learning prefix must use FNP3'
                    pool=sorted(pool,key=lambda x:(x.col,x.row,x.id));pool=[x for x in pool if(x.col,x.row,x.id)not in fixed_used[1-role]]
                print('MAP_REQUEST',role,prefix,str(type),num_nps,len(pool),flush=True)
                assert len(pool)>=num_nps,(str(type),num_nps,len(pool))
                selected=pool[:num_nps];mapping.append(dict(role=role,prefix=prefix,type=str(type),requested=num_nps,selected=[[x.col,x.row,x.id]for x in selected]));return selected
        device=akida.AKD1500();m.map(device,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(device));assert len(m.sequences)==len(m.sequences[0].passes)==1
        if not prefix:
            for row in mapping:
                if row['role']==role and not row['prefix']and row['type'].endswith('FNP2'):fixed_used[role].update(map(tuple,row['selected']))
        return bytes(m.sequences[0].program)
    def model(k,competition):
        n,u=dims[k];m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));m.add(akida.FullyConnected(name=f'plastic{k}',units=u,weights_bits=1,activation=True,act_bits=1));m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=2 if k==0 else 1,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=competition));layer=m.layers[-1]
        for name,value in [('weights',ws[k].reshape(1,1,n,u)),('threshold_learning',lts[k]),('threshold',ts[k]),('act_step',np.ones(u,'f4'))]:layer.set_variable(name,value)
        return m
    for k,(n,u)in enumerate(dims):
        targets=TARGETS[:5]if k==0 else TARGETS[5:7];m=model(k,.25 if k==0 else 1/16);prefix=mapped(m,targets,k,True);m.map(None);entries=g.matrices()if k==0 else[(np.eye(8,dtype='i1'),np.zeros(8,'i4'))]
        for j,(w,t)in enumerate(entries):
            raw=j==len(entries)-1;m.add(akida.FullyConnected(name=f'fixed{k}_{j}',units=w.shape[1],weights_bits=2 if k==0 and j==0 else 1,activation=not raw,act_bits=1));layer=m.layers[-1];layer.set_variable('weights',w.reshape(1,1,*w.shape))
            if not raw:layer.set_variable('threshold',t);layer.set_variable('act_step',np.ones(len(t),'f4'))
        image,audit=assemble(prefix,mapped(m,targets,k),n,u);images.append(image);h,d=read(image);lr=h['learning'];bank={name:words(d,lr[key])for name,key in [('infer','inference'),('learn','learning')]};bank['ram']=words(d,lr['ram'][0]);bank['np']=lr['np'];banks.append(bank)
        assert bank['np']in targets
        for name in ['ram','infer','learn']:np.array(bank[name],'<u4').tofile(out/f'bank{k}.{name}')
        ts[k].tofile(out/f'bank{k}.threshold')
        if k==1:
            m2=model(k,31/32);h2,d2=read(mapped(m2,targets,k,True));retire_prefix=words(d2,h2['learning']['learning']);ph,pd=read(prefix);normal_prefix=words(pd,ph['learning']['learning']);changes=[i for i,(a,b)in enumerate(zip(normal_prefix,retire_prefix))if a!=b];assert changes==[38],changes
            # Assemble routes belong to the fixed-readout application, not the
            # single-layer prefix. Transfer only the qualified competition word.
            retire=list(bank['learn']);retire[38]=retire_prefix[38];assert [i for i,(a,b)in enumerate(zip(bank['learn'],retire))if a!=b]==[38];np.array(retire,'<u4').tofile(out/'bank1.retire')
    image=save_roles(out,images);fixed=[p for npid,p in fixed_packets(image)if npid not in[b['np']for b in banks]];assert len(fixed)==4
    for j,p in enumerate(fixed):np.array(p,'<u4').tofile(out/f'fixed{j}.ram')
    (out/'fixed_count.txt').write_text('4');(out/'normal_cap.txt').write_text('30')
    for j,(w,t)in enumerate(g.matrices()):w.tofile(out/f'fixed_layer{j}.weights');t.tofile(out/f'fixed_layer{j}.threshold')
    (out/'geometry.txt').write_text(f'{g.n0} {g.u0} {g.n1} 8 {g.W} {g.P}\n');(out/'LAYOUT.json').write_text(json.dumps(g.__dict__,indent=2)+'\n')
    result=dict(status='COMPILER_GEOMETRY_PASS',profile=profile,source_inputs=g.n0,source_units=g.u0,content_inputs=g.n1,payload=g.P,output_words=g.W,mapping=mapping,learners_FNP3=True,fixed_FNP2_allowed_only_in_disjoint_role_partitions=True,physical_device_opened=False,program_sha256=hashlib.sha256(image).hexdigest(),SDK_version=akida.__version__);(out/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('profile',choices=['quad13','oct9']);a=p.parse_args();prepare(a.out,a.profile)
