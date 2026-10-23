"""Bounded recorded-Pong cost comparison using existing verified transport tools.

This study is independent of the failed learning gate. It preserves the exact
program, input/label stream, output scores and final learning state. Queued
results concern a prerecorded workload, not causal online game batching.
"""
from pathlib import Path
import argparse,datetime,fcntl,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pong.experiment import save,sha
from verify import verify


def preflight(study):
    authority=json.loads((study/'AUTHORITY.json').read_text());state=json.loads((study/'STATE.json').read_text())
    assert not state['research_admissions_closed']
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(authority['scientific_cutoff_utc'])
    freeze=json.loads((study/'evidence/PHYSICAL_SOURCE_FREEZE.json').read_text())
    for p,h in freeze['hashes'].items():assert sha(ROOT/p)==h,('source drift',p)
    assert len(list((study/'raw').glob('*/OWNER.json')))<authority['maximum_physical_sessions']


def profile():
    paths=['/proc/loadavg','/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq']
    return {p:Path(p).read_text().strip() for p in paths if Path(p).exists()}


def smoke(study):
    preflight(study);out=study/'raw/transport_smoke';out.mkdir(exist_ok=False)
    fixture=study/'fixtures-replay01/pong';rec=dict(name='transport_smoke',samples=0,native_forwards=0,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary_sha256=sha(ROOT/'build/offload_bench'))
    save(out/'OWNER.json',rec)
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (out/'native.stdout').open('w') as stdout,(out/'native.stderr').open('w') as stderr:
            child=subprocess.Popen(['taskset','-c','2',str(ROOT/'build/offload_bench'),'transport',str(fixture),str(out/'native'),'1','0','1','0'],stdout=stdout,stderr=stderr)
            rec['native_pid']=child.pid;save(out/'OWNER.json',rec)
            try:code=child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:code=child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
        rec.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,
            device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        if code==0:rec['results']=json.loads((out/'native.stdout').read_text())
        save(out/'RECEIPT.json',rec)
        assert code==0 and rec['device_released'],(out/'native.stderr').read_text()
    print(json.dumps(rec))


def round_run(study,index):
    preflight(study);plan=json.loads((study/'PROTOCOL.json').read_text());assert index in [0,1,2]
    assert json.loads((study/'raw/transport_smoke/RECEIPT.json').read_text())['status']=='PASS'
    if index:assert json.loads((study/'evidence'/f'ROUND_{index-1}.json').read_text())['status']=='PASS'
    fixture=study/'fixtures-replay01/pong';env=os.environ.copy();env['AKD_LAB_STUDY']=str(study.relative_to(ROOT))
    rows=[]
    for kind in plan['orders'][index]:
        preflight(study);c=plan['configurations'][kind];name=f'r{index}_{kind}';before=profile()
        args=[sys.executable,str(ROOT/'tools/run_cell.py'),name,str(fixture.relative_to(ROOT)),
            '--samples',str(plan['samples']),'--learning','1','--cpu-repeats',str(plan['cpu_repeats']),
            '--bench','offload_bench','--wait-us',str(c['wait_us']),
            '--dma-read-min',str(c['dma_read_min']),'--dma-write-min',str(c['dma_write_min'])]
        if c['resident']:args.append('--resident')
        subprocess.run(args,env=env,check=True,stdout=subprocess.DEVNULL)
        raw=study/'raw'/name;checked=verify(ROOT,raw)
        # Directly bind to the ORIGINAL sequential physical learning result.
        saved=study/'reference'
        assert (raw/'native.scores').read_bytes()==(saved/'native.scores').read_bytes()
        assert (raw/'native.ram').read_bytes()==(saved/'native.ram').read_bytes()
        receipt=json.loads((raw/'RECEIPT.json').read_text())
        row=dict(name=name,configuration=kind,verification=checked,profile_before=before,profile_after=profile(),results=receipt['results'])
        rows.append(row);save(study/'evidence'/f'CELL_{name}.json',row)
        native=receipt['results']['native'];cpu=receipt['results']['cpu']
        print(json.dumps(dict(name=name,status='EXACT_TASK_REPLAY_PASS',native_us=native['wall_ns']/plan['samples']/1000,
            native_thread_cpu_us=native['thread_cpu_ns']/plan['samples']/1000,
            exact_cpu_us=cpu['wall_ns']/plan['samples']/plan['cpu_repeats']/1000)),flush=True)
    save(study/'evidence'/f'ROUND_{index}.json',dict(status='PASS',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('phase',choices=['smoke','0','1','2']);a=p.parse_args()
    smoke(a.study.resolve()) if a.phase=='smoke' else round_run(a.study.resolve(),int(a.phase))
