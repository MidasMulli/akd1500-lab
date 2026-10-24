"""Reject changes outside the known parameter packets before physical admission."""
from pathlib import Path
import hashlib
import json


def validate(study):
    fixtures=study/'fixtures-ablation02';base=(fixtures/'baseline.bin').read_bytes()
    rows=[]
    for name in ['suppress_A','suppress_B','suppress_both']:
        actual=(fixtures/(name+'.bin')).read_bytes();rebuilt=bytearray(base);changed=[]
        for j in range(5):
            old=(fixtures/f'baseline.packet{j}.ram').read_bytes()
            new=(fixtures/f'{name}.packet{j}.ram').read_bytes()
            assert len(old)==len(new)
            at=base.find(old);assert at>=0 and base.find(old,at+1)<0,('ambiguous packet',j)
            rebuilt[at:at+len(new)]=new
            if old!=new:changed.append(j)
        assert bytes(rebuilt)==actual,('change outside known parameter packets',name)
        rows.append(dict(name=name,changed_packets=changed,
            entire_program_accounted_for=True,program_sha256=hashlib.sha256(actual).hexdigest()))
    result=dict(status='PASS',checks=rows)
    (study/'evidence/FULL_PROGRAM_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);print(json.dumps(validate(p.parse_args().study)))
