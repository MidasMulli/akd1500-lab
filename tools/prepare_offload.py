"""Dense/event-input fixtures with a guarded SDK-2.19.3 ProgramInfo adapter."""
from pathlib import Path
import argparse,copy,hashlib,json,struct
import numpy as np
from program_codec import read,build

def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    info,data=read(source.with_suffix('.bin').read_bytes());assert info['s4']==0 and len(info['passes'])==1 and info['learning']
    n,u,nw,classes=map(int,source.with_suffix('.meta').read_text().split());assert n==512 and u==512
    rows=[]
    for active in [32,128]:
        rng=np.random.default_rng(84101+active);patterns=np.zeros((16,n),np.uint8)
        for row in patterns:row[rng.choice(n,active,replace=False)]=1
        x=patterns[np.arange(1024)%16];labels=np.array([i%2 for i in range(1024)],dtype='<i4')
        for sparse in [False,True]:
            name=f'active{active}_'+('sparse' if sparse else 'dense');h=copy.deepcopy(info)
            if sparse:h['s4']=1 # reviewed generated schema: IoType_fnp_sparse
            image=build(h,data);hh,dd=read(image);assert hh==h and dd==data
            assert {k:v for k,v in h.items() if k!='s4'}=={k:v for k,v in info.items() if k!='s4'}
            for ext,b in [('bin',image),('ram',source.with_suffix('.ram').read_bytes()),('meta',source.with_suffix('.meta').read_bytes()),('output',source.with_suffix('.output').read_bytes()),('inputs',x.tobytes()),('labels',labels.tobytes())]:(out/(name+'.'+ext)).write_bytes(b)
            rows.append(dict(name=name,status='OFFLINE_ADAPTER_VALIDATED',inputs=n,neurons=u,active_input_features=active,sparse=sparse,payload_bytes_per_input=active*8 if sparse else n,all_configuration_tracks_unchanged=True,only_ProgramInfo_field_changed='input_type: dense(0) to fnp_sparse(1)' if sparse else None,program_sha256=hashlib.sha256(image).hexdigest()))
    (out/'OFFLOAD.json').write_text(json.dumps(dict(status='OFFLINE_ONLY',physical_device_opened=False,cells=rows),indent=2)+'\n');print(json.dumps(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
