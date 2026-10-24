"""Require semantic rejection after validly rehashing corrupted evidence."""
from pathlib import Path
import copy,hashlib,json,tempfile
import numpy as np
from transition import archive as a
def main():
    original=a.BASE;a.audit();meta=json.loads(original.with_suffix('.json').read_text())
    with np.load(original.with_suffix('.npz'),allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    rn='native_s110101_native_h32';fn='native_s110101_frozen_h32_attempt2';cases=[('returned_query',rn,'scores',24*256),('visible_weight',rn,'states',16*98+6),('returned_trajectory',rn,'rollout',256),('dependency_pointer',rn,'descriptors',17),('quiet_read',rn,'quiet',3),('frozen_memory',fn,'ram',6)]
    rows=[]
    try:
      with tempfile.TemporaryDirectory(prefix='akd-audit-mutations-') as tmp:
        for label,run,key,index in cases:
            arrays={k:v.copy() for k,v in source.items()};m=copy.deepcopy(meta);info=m['runs'][run]['arrays'][key];value=arrays[info['key']];value[index]^=1;info['sha256']=hashlib.sha256(value.tobytes()).hexdigest();a.BASE=Path(tmp)/label;np.savez_compressed(a.BASE.with_suffix('.npz'),**arrays);m['archive_sha256']=a.sha(a.BASE.with_suffix('.npz'));a.save(a.BASE.with_suffix('.json'),m)
            try:a.audit()
            except AssertionError:rows.append(dict(mutation=label,status='SEMANTIC_REJECTION_PASS',outer_and_array_hashes_valid=True))
            else:raise RuntimeError('audit accepted corrupted '+label)
    finally:a.BASE=original
    result=dict(status='PASS',archive_sha256=meta['archive_sha256'],mutations=rows);a.save(original.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
