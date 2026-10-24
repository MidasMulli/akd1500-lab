"""Independent integer/bitset check of one raw physical ablation before expansion."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from metal.ablation import P, S


def check(name):
    d=S/'raw'/name;r=json.loads((d/'RECEIPT.json').read_text());assert r['status']=='PASS' and r['device_released']
    for f,h in r['hashes'].items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==h
    f=P/r['fixture'];meta=json.loads(f.with_suffix('.json').read_text())
    rng=np.random.default_rng(98120)
    weights=[sum(1<<int(k) for k in rng.choice(512,32,replace=False)) for j in range(256)]
    x=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(64,512)
    actual=np.fromfile(d/'native.scores','<i4').reshape(64,2)
    for i,values in enumerate(x):
        bits=sum(1<<int(k) for k in np.flatnonzero(values))
        responses=[int((bits&w).bit_count()>meta['thresholds'][j]) for j,w in enumerate(weights)]
        assert actual[i].tolist()==[sum(responses[:128]),sum(responses[128:])],('integer native recount',name,i)
    for j,packet in enumerate(meta['packets']):
        expected=(f.parent/f'{name}.packet{j}.ram').read_bytes()
        for when in ['initial','final']:
            observed=(d/f'native.packet{j}.{when}').read_bytes()
            assert observed==expected,('physical memory',name,j,when)
            assert hashlib.sha256(observed).hexdigest()==packet['sha256']
    result=dict(status='INDEPENDENT_INTEGER_AND_MEMORY_PASS',name=name,native_forwards=64,
                native_output_values=128,physical_regions_checked=5,initial_and_final_readbacks=10)
    (d/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');print(json.dumps(check(p.parse_args().name)))
