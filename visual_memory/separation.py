"""Resident disjoint V1 chains and raw on-card mailbox, source-only builders."""
from pathlib import Path
import argparse,copy,json,struct
import numpy as np
from visual_memory.data import save,sha
from tools.program_codec import read,build,words
TARGETS=[[1,3,0],[2,1,0],[2,2,0],[2,3,0],[3,1,0],[3,2,0],[3,3,0]]

def mapped(m,targets):
    import akida
    class Select(akida.MapConstraints):
        def select_nps(self,source_nps,num_nps,type):
            if type==akida.NP.Type.FNP2:raise RuntimeError('FNP2 is not admitted')
            if type==akida.NP.Type.FNP3:
                q=[x for t in targets for x in source_nps if [x.col,x.row,x.id]==t];assert len(q)>=num_nps;return q[:num_nps]
            return list(source_nps)[:num_nps]
    d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
    return bytes(m.sequences[0].program)

def packets(image):
    h,d=read(image);result=[]
    for rec in h['passes'][0]:
        for span in rec:
            a=words(d,span);i=0
            while i<len(a):
                head,size=a[i:i+2];n=(size>>16)&16383
                if size>>31:n*=16
                v=a[i:i+n+2];assert len(v)==n+2
                mask=(head>>8)&15
                result.append(dict(header=head,uid=head&15,np=[head>>24,(head>>16)&255,mask.bit_length()-1] if mask and not(mask&(mask-1)) else None,words=v))
                i+=n+2
    return result

def union(a,b):
    ha,da=read(a);hb,db=read(b);assert ha['device']==hb['device'] and len(ha['passes'])==len(hb['passes'])==1
    h=copy.deepcopy(ha);data=bytearray(da);shift=len(data)-4;data.extend(db[4:]);h['passes'][0].extend([[[o+shift,n] for o,n in rec] for rec in hb['passes'][0]])
    if hb['learning']:
        # Append the second learner's initial inference parameters and full RAM.
        lr=hb['learning'];h['passes'][0].append([[lr['inference'][0]+shift,lr['inference'][1]],*[ [o+shift,n] for o,n in lr['ram'] ]])
    struct.pack_into('<I',data,0,len(data)-4);out=build(h,bytes(data));hh,dd=read(out);assert hh==h and dd==bytes(data)
    for rec in hb['passes'][0]:
        for o,n in rec:assert words(db,[o,n])==words(dd,[o+shift,n])
    return out

def save_roles(out,images):
    meta=[]
    for k,image in enumerate(images):
        (out/f'role{k}.bin').write_bytes(image);h,d=read(image);pp=packets(image)
        ingress=[p['words'] for p in pp if p['np']==[0,0,0] or p['np'] is None]
        assert all(v[0]==0x88 and len(v)==3 and (v[1]&65535) in [4,64,68] for v in ingress), 'New shared ingress registers require review'
        (out/f'role{k}.ingress_count').write_text(str(len(ingress)))
        for j,v in enumerate(ingress):np.array(v,'<u4').tofile(out/f'role{k}.ingress{j}')
        np.array(h['epg'],'<u4').tofile(out/f'role{k}.epg')
        meta.append(dict(input_dims=h['input_dims'],output_dims=h['output_dims'],epg=h['epg'],ingress_packets=len(ingress),packet_index=[{key:val for key,val in p.items() if key!='words'}|dict(count=len(p['words'])) for p in pp]))
    image=union(*images);(out/'model.bin').write_bytes(image)
    from visual_memory.prepare_v4 import fixed_packets
    fixed=fixed_packets(image)
    for j,(_,v) in enumerate(fixed):np.array(v,'<u4').tofile(out/f'fixed{j}.ram')
    (out/'fixed_count.txt').write_text(str(len(fixed)))
    save(out/'ROLES.json',meta)
    return image

def prepare_probe(out):
    import akida
    out.mkdir(exist_ok=False);images=[]
    for k,(n,u,target) in enumerate([(1024,256,TARGETS[0]),(256,8,TARGETS[4])]):
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));m.add(akida.FullyConnected(name=f'role{k}',units=u,weights_bits=1,activation=False));w=np.zeros((n,u),'i1')
        for j in range(u):
            if k or not 240<=j<248:w[4*j,j]=1
        m.layers[-1].set_variable('weights',w.reshape(1,1,n,u));x=np.random.default_rng(27102751+k).integers(2,size=(32,n),dtype='u1');np.testing.assert_array_equal(m.forward(x.reshape(-1,1,1,n)).reshape(-1,u),x.astype('i4')@w.astype('i4'));images.append(mapped(m,[target]));m.map(None)
    image=save_roles(out,images);rng=np.random.default_rng(27102751);mail=rng.integers(2,size=(16,256),dtype='u4');mail[:,240:256]=0;mail.astype('<u4').tofile(out/'observations.bin')
    save(out/'BUILD.json',dict(status='VIRTUAL_AND_UNION_PASS',physical_device_opened=False,rows=16,expected_forwards=64,program_sha256=sha(out/'model.bin'),content_bytes=[0,256],mailbox_bytes=1024,recall_header_byte_range=[960,992],recall_values_byte_range=[992,1024],CPU_intermediate_data_copy=False,native_branch_merge_required=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();prepare_probe(a.out)
