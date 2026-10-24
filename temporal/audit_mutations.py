"""Semantic negative checks with valid outer and per-array hashes."""
from pathlib import Path
import copy,hashlib,json,tempfile
import numpy as np
from temporal import archive as a
def main():
    original=a.BASE;a.audit();meta=json.loads(original.with_suffix('.json').read_text())
    with np.load(original.with_suffix('.npz'),allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    run='native_s112101_native_e8';cases=[('returned_code','scores',9*256,1),('learning_mask','states',9*98+6,1),('native_history_pointer','input_addresses',9,1),('teacher_class','descriptors',9*16+8,1<<16),('dwell','delays_ns',25,1<<60)]
    rows=[]
    try:
      with tempfile.TemporaryDirectory(prefix='akd-temporal-mutations-') as tmp:
        for label,key,index,bit in cases:
            arrays={k:v.copy() for k,v in source.items()};m=copy.deepcopy(meta);info=m['runs'][run]['arrays'][key];value=arrays[info['key']]
            if label=='dwell':value[index]=0
            else:value[index]^=bit
            info['sha256']=hashlib.sha256(value.tobytes()).hexdigest();a.BASE=Path(tmp)/label;np.savez_compressed(a.BASE.with_suffix('.npz'),**arrays);m['archive_sha256']=a.sha(a.BASE.with_suffix('.npz'));a.save(a.BASE.with_suffix('.json'),m)
            try:a.audit()
            except AssertionError:rows.append(dict(mutation=label,status='SEMANTIC_REJECTION_PASS'))
            else:raise RuntimeError('accepted corrupted '+label)
    finally:a.BASE=original
    result=dict(status='PASS',archive_sha256=meta['archive_sha256'],mutations=rows);a.save(original.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
