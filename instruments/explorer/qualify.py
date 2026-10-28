"""Prospective gates: packed CPU first, then a finite new physical trial."""
import argparse
import json
from pathlib import Path
from admission import learned_policy as model
from instruments.explorer.live import Resident, action_rows, ROOT
from instruments.explorer.data import sha


def qualify(mode):
    results=[]
    arms=model.ARMS if mode=='cpu' else ['native','manager_frozen','open']
    for arm in arms:
        s=Resident(mode,arm)
        try:
            for e in model.schedule('pilot' if mode=='cpu' else 'support',arm,0):s.step(e)
            # Interaction must preserve query-only RAM, block untrusted learning,
            # learn reliability from outcomes, and then admit a memory revision.
            if mode=='native':
                def act(a):
                    return [s.step(e) for e in action_rows(a,arm,len(s.stepper.rows))]
                first=act(dict(action='query',key=0))[-1]
                assert first['memory_flags']==[1,0]
                bad=act(dict(action='propose',key=0,policy=1,candidate=1))
                if arm!='open':assert all(not any(c['bank'] in [1,2] for c in e['changes']) for e in bad)
                act(dict(action='feedback',policy=1,candidate=1,outcome=1))
                act(dict(action='propose',key=0,policy=1,candidate=1))
                last=act(dict(action='query',key=0))[-1]
                assert last['memory_flags']==([1,0] if arm=='manager_frozen' else [0,1])
                for _ in range(3):assert not act(dict(action='query',key=0))[-1]['changes']
                assert act(dict(action='query',key=2))[-1]['memory_flags']==[1,0]
            s.close();assert s.manifest['status']=='CLOSED_VERIFIED',s.manifest
            results.append(dict(path=str(s.path),mode=mode,arm=arm,events=s.manifest['events'],
                                receipt_sha256=sha(s.path/'RECEIPT.json')))
        except BaseException:s.close(failed=True);raise
    out=dict(status='NATIVE_QUERY_TEACH_PASS' if mode=='native' else 'PACKED_CPU_PILOT_PASS',
             mode=mode,results=results,native_binary_sha256=sha(ROOT/'build/explorer_native'),
             source_hashes={name:sha(Path(__file__).parent/name) for name in ['data.py','live.py','native.cpp','qualify.py','build-native.sh']},
             scope='Bounded resident sessions with exact Python/packed CPU checks; no electrical spike or efficiency measurement')
    path=Path(__file__).parent/'generated'/('NATIVE_QUALIFICATION.json' if mode=='native' else 'CPU_QUALIFICATION.json')
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','native']);a=p.parse_args()
    if a.mode=='native':
        gate=json.loads((Path(__file__).parent/'generated/CPU_QUALIFICATION.json').read_text())
        assert gate['status']=='PACKED_CPU_PILOT_PASS'
        assert gate['native_binary_sha256']==sha(ROOT/'build/explorer_native')
        for name,digest in gate['source_hashes'].items():assert sha(Path(__file__).parent/name)==digest
    qualify(a.mode)
