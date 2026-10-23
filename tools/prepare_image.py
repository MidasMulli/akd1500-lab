"""Prepare a fresh explicit image-service study; never open the physical device."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
P=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();s=a.study.resolve()
    auth=json.loads((s/'AUTHORITY.json').read_text());assert not (s/'FROZEN_MANIFEST.json').exists()
    for name in ['service','sources']:shutil.copytree(P/'image_service'/name,s/name)
    for name in ['verify_native.py','verify_service.py','run_monitored.py','verify_recovery.py','run_recovery_monitored.py']:shutil.copy2(P/'image_service'/name,s/name)
    auth['maximum_physical_AKD_sessions']=auth.get('maximum_physical_sessions',8);auth['shared_device_lock']='/tmp/akd1500-research.lock'
    (s/'AUTHORITY.json').write_text(json.dumps(auth,indent=2)+'\n')
    for name in ['runs','cpu_runs','evidence']:(s/name).mkdir(exist_ok=True)
    (s/'STATE.json').write_text(json.dumps(dict(status='PREPARED_NOT_ADMITTED',research_admissions_closed=False))+'\n')
    (s/'ADMISSION.json').write_text(json.dumps(dict(watch=False,smoke=False,confirm=False))+'\n')
    subprocess.run([sys.executable,str(s/'sources/prepare.py')],check=True)
    sys.path.insert(0,str(s/'service'));from artifacts import fixtures
    for seed in [71011,71012,71013]:fixtures(s/'inputs'/str(seed),seed)
    shutil.copy2(P/'image_service/FORECAST.json',s/'evidence/PRE_PHYSICAL_FORECAST.json')
    shutil.copy2(P/'build/image_service',s/'native_service')
    files={str(f.relative_to(s)):hashlib.sha256(f.read_bytes()).hexdigest() for f in s.rglob('*') if f.is_file() and f.parts[-2] not in ['evidence'] and f.name not in ['STATE.json','ADMISSION.json','FROZEN_MANIFEST.json'] and '__pycache__' not in f.parts}
    (s/'FROZEN_MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n');print('PREPARED; review compiler output before setting watch admission')
if __name__=='__main__':main()
