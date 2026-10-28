from pathlib import Path
import argparse,hashlib,json
import numpy as np

p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('prefix',type=Path);p.add_argument('--native',action='store_true');a=p.parse_args();f=a.fixture;z=np.load(f/'TRACE.npz');count=len(z['outputs']);g=list(map(int,(f/'geometry.txt').read_text().split()));n0,u0,n1,u1,W,P,output1=g
for key in ['outputs','states0','states1']:assert np.array_equal(np.fromfile(str(a.prefix)+'.'+key,'<u4').reshape(z[key].shape),z[key]),key
result=dict(status='PASS',forwards=count,complete_outputs_and_both_RAM_exact=True,physical=a.native)
if a.native:
    rows=json.loads((f/'ROWS.json').read_text());desc=np.fromfile(str(a.prefix)+'.descriptors','<u4').reshape(count,16);completed=np.fromfile(str(a.prefix)+'.completed_indices','<u4');assert np.array_equal(completed,np.arange(count));previous=None;feedback=0
    for i,r in enumerate(rows):
        src,dst=int(desc[i,1]),int(desc[i,7])
        if r['net']==0:
            if not r['external']:assert src==previous+4*P;feedback+=1
            previous=dst+32
        else:assert src==previous and dst==previous+4*(P+240);feedback+=1
    for j in range(5):assert Path(str(a.prefix)+f'.fixed{j}.initial').read_bytes()==Path(str(a.prefix)+f'.fixed{j}.final').read_bytes()
    for role in [0,1]:
        width=n0 if role==0 else n1;indices=[i for i,r in enumerate(rows)if r['net']==role];actual=np.fromfile(str(a.prefix)+f'.actual_inputs{role}','u1').reshape(len(indices),width);expected=np.fromfile(f/'inputs.bin','u1').reshape(count,1024)[indices,:width]
        if role==0:assert np.array_equal(actual[:,:960],expected[:,:960])and np.array_equal(actual[:,992:],expected[:,992:])
        else:assert np.array_equal(actual,expected)
    result.update(native_feedback_windows_verified=feedback,fixed_RAM_unchanged=True,all_actual_defined_inputs_exact=True,observed=count)
else:
    result['fixture_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in f.iterdir()if p.is_file()and p.name!='CPU_QA.json'}
path=Path(str(a.prefix)+'.QA.json')if a.native else f/'CPU_QA.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='fixture_hashes'},indent=2))
