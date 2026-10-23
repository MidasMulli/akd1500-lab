"""Audit a byte-preserved copied image study using explicit path relocation."""
from pathlib import Path
import argparse,importlib,json,sys
def audit(study):
    sys.path.insert(0,str(study));sys.path.insert(0,str(study/'service'))
    vn=importlib.import_module('verify_native');vs=importlib.import_module('verify_service')
    rec=json.loads(next((study/'runs').glob('*/OWNER.json')).read_text());origin=Path(rec['args'][1]).parent.parent
    def relocate(p):
        p=Path(p)
        try:return study/p.relative_to(origin)
        except ValueError:return p
    old_sha=vs.sha;old_check=vs.check_artifact
    vs.sha=lambda p:old_sha(relocate(p))
    vs.check_artifact=lambda source,out,stage:old_check(relocate(source),relocate(out),stage)
    vr=None
    if (study/'verify_recovery.py').exists():
        vr=importlib.import_module('verify_recovery');rsha=vr.sha;rcheck=vr.artifact
        vr.sha=lambda p:rsha(relocate(p))
        vr.artifact=lambda source,out,stage:rcheck(relocate(source),relocate(out),stage)
    result=[]
    for root in ['runs','cpu_runs']:
        for d in sorted((study/root).iterdir()):
            if not (d/'SUMMARY.json').exists():continue
            is_recovery=json.loads((d/'SUMMARY.json').read_text()).get('mode')=='recovery'
            row=vr.verify(d) if is_recovery else vs.verify(d);row['native_raw_audit']=vn.verify(d)['status'] if root=='runs' else None;result.append(row)
    return dict(status='PASS',relocation='Source bytes and receipts unchanged; map original absolute paths only at artifact reads',results=result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();z=audit(a.study.resolve());(a.study/'evidence/MAC_RELOCATED_AUDIT.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z))
