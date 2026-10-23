"""Bounded native adapter. It never selects an action or writes learned connections."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,select,subprocess,time

P=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
class NativeClient:
    def __init__(self,name,stage,policy='twohop'):
        authority=json.loads((P/'AUTHORITY.json').read_text());assert not json.loads((P/'STATE.json').read_text()).get('research_admissions_closed',False),'study closed'
        assert json.loads((P/'ADMISSION.json').read_text()).get(stage) is True
        now=datetime.datetime.now(datetime.timezone.utc);assert now<datetime.datetime.fromisoformat(authority['scientific_cutoff_utc'])
        for f,h in json.loads((P/'FROZEN_MANIFEST.json').read_text()).items():assert sha(P/f)==h,f
        assert len(list((P/'runs').glob('*/OWNER.json')))<authority['maximum_physical_AKD_sessions']
        self.lock=open(authority['shared_device_lock'],'a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        self.max_seconds=min(180,authority.get('maximum_individual_seconds',180));self.d=P/'runs'/name;self.d.mkdir(exist_ok=False);self.started=time.monotonic();self.buffer=b'';self.counter=0;self.closed=False
        self.stdout=(self.d/'stdout.jsonl').open('w');self.commands=(self.d/'commands.txt').open('w');self.stderr=(self.d/'stderr.txt').open('w')
        args=[str(P/'native_service'),str(P/'frozen'/f'{policy}.bin'),str(self.d/'mmio.csv'),str(P/'frozen'/f'{policy}_PLAN.txt'),
              str(P/'frozen'/f'{policy}_m0_inference.regs'),str(P/'frozen'/f'{policy}_m0_learning.regs'),str(P/'frozen'/f'{policy}_m1_inference.regs')]
        self.rec=dict(name=name,stage=stage,policy=policy,start=now.isoformat(),args=args,program=f'frozen/{policy}.bin',
                      audit=f'frozen/{policy}_AUDIT.json',program_sha256=sha(P/f'frozen/{policy}.bin'),native_sha256=sha(P/'native_service'),
                      manifest_sha256=sha(P/'FROZEN_MANIFEST.json'),wait_us=int(os.environ.get('AKD_IMAGE_WAIT_US','50')))
        self.proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
        self.rec['pid']=self.proc.pid;save(self.d/'OWNER.json',self.rec)
        try:self.ready=self.read_until('ready')
        except BaseException:self.close(failed=True);raise
    def read_until(self,kind,timeout=15):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if b'\n' not in self.buffer:
                readable,_,_=select.select([self.proc.stdout],[],[],max(0,deadline-time.monotonic()))
                if not readable:break
                block=os.read(self.proc.stdout.fileno(),1048576)
                if not block:raise RuntimeError('native exited: '+(self.d/'stderr.txt').read_text()[-2000:])
                self.buffer+=block
            while b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1);obj=json.loads(line);self.stdout.write(line.decode()+'\n');self.stdout.flush()
                if obj['kind']==kind:return obj
        raise TimeoutError('waiting for native '+kind)
    def send(self,command,kind):
        assert time.monotonic()-self.started<self.max_seconds,'native session time cap'
        self.commands.write(command+'\n');self.commands.flush();self.proc.stdin.write((command+'\n').encode());self.proc.stdin.flush();return self.read_until(kind)
    def query(self):
        self.counter+=1;return self.send(f'Q {self.counter}','chain')
    def observe(self,outcome):
        self.counter+=1;return self.send(f'O {self.counter} {outcome}','chain')
    def snapshot(self):
        self.counter+=1;return self.send(f'S {self.counter}','snapshot')
    def request_start(self):
        self.counter+=1;return self.send(f'R {self.counter}','request_start')
    def mode(self,transition=1,history=1):return self.send(f'M {transition} {history}','mode')
    def close(self,failed=False):
        if self.closed:return
        self.closed=True
        try:
            if self.proc.poll() is None:
                try:
                    self.commands.write('EXIT\n');self.commands.flush();self.proc.stdin.write(b'EXIT\n');self.proc.stdin.flush();self.read_until('final',timeout=8)
                except Exception:
                    self.proc.stdin.close()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.rec.update(exit_code=self.proc.returncode,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),failed_at_host=failed)
        finally:
            self.stdout.close();self.commands.close();self.stderr.close();self.rec['commands_sha256']=sha(self.d/'commands.txt');save(self.d/'RECEIPT.json',self.rec);self.lock.close()
