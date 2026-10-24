"""Require semantic rejection after validly rehashing corrupted evidence."""
from pathlib import Path
import copy,hashlib,json,tempfile
import numpy as np
from plasticity import archive as a
def main():
    original=a.BASE;a.audit();meta=json.loads(original.with_suffix('.json').read_text())
    with np.load(original.with_suffix('.npz'),allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    rn='native_s111101_native_e12';fn='native_s111101_frozen_e12';cases=[('returned_query',rn,'scores',25*256),('visible_weight',rn,'states',17*98+6),('carried_state_pointer',rn,'input_addresses',29),('teacher_descriptor',rn,'descriptors',17*16+8),('quiet_read',rn,'quiet',5),('frozen_memory',fn,'ram',6)]
    rows=[]
    try:
      with tempfile.TemporaryDirectory(prefix='akd-audit-mutations-') as tmp:
        for label,run,key,index in cases:
            arrays={k:v.copy() for k,v in source.items()};m=copy.deepcopy(meta);info=m['runs'][run]['arrays'][key];value=arrays[info['key']]
            # Teacher classes occupy bits 16..25; low descriptor bits are not labels.
            value[index]^=(1<<16) if label=='teacher_descriptor' else 1
            info['sha256']=hashlib.sha256(value.tobytes()).hexdigest();a.BASE=Path(tmp)/label;np.savez_compressed(a.BASE.with_suffix('.npz'),**arrays);m['archive_sha256']=a.sha(a.BASE.with_suffix('.npz'));a.save(a.BASE.with_suffix('.json'),m)
            try:a.audit()
            except AssertionError:rows.append(dict(mutation=label,status='SEMANTIC_REJECTION_PASS',outer_and_array_hashes_valid=True))
            else:raise RuntimeError('audit accepted corrupted '+label)
    finally:a.BASE=original
    result=dict(status='PASS',archive_sha256=meta['archive_sha256'],mutations=rows);a.save(original.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
