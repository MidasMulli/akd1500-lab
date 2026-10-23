"""Independent observer of owned worker process states; no acting input to controller."""
from pathlib import Path
import json,subprocess,sys,time
P=Path(__file__).resolve().parent
name,stage,seed,arm=sys.argv[1:];base='cpu_runs' if arm=='cpu' else 'runs';d=P/base/name
pr=subprocess.Popen([sys.executable,str(P/'service/app.py'),'series',name,stage,seed,arm]);monitor=None;owners=None
try:
    while pr.poll() is None:
        if (d/'WORKER_OWNERS.json').exists():
            if monitor is None:monitor=(d/'PROCESS_STATES.jsonl').open('w');owners=json.loads((d/'WORKER_OWNERS.json').read_text())
            states={}
            for key,pid in owners.items():
                try:states[key]=next(l.split()[1] for l in Path(f'/proc/{pid}/status').read_text().splitlines() if l.startswith('State:'))
                except (FileNotFoundError,StopIteration):states[key]='gone'
            monitor.write(json.dumps(dict(unix_time=time.time(),states=states))+'\n');monitor.flush()
        time.sleep(.02)
    raise SystemExit(pr.returncode)
finally:
    if monitor:monitor.close()
    if pr.poll() is None:
        pr.terminate()
        try:pr.wait(timeout=12)
        except subprocess.TimeoutExpired:pr.kill();pr.wait()
