"""Four dedicated owned workers; faults use actual SIGSTOP, never fake outcome labels."""
from pathlib import Path
import json,os,signal,socket,subprocess,sys,time

P=Path(__file__).resolve().parent.parent
class Workers:
    def __init__(self,d):
        self.d=Path(d);self.sockets={};self.processes={};self.logs=[]
        try:self._start()
        except BaseException:self.close();raise
    def _start(self):
        for stage in ['decode','export']:
            for backend in ['cpu','gstreamer']:
                key=(stage,backend);short=f'{stage}_{backend}';sock=Path('/tmp')/f'akimg_{os.getpid()}_{short}.sock';assert not sock.exists()
                out=(self.d/(short+'.log')).open('w');self.logs.append(out)
                proc=subprocess.Popen([sys.executable,str(P/'service/worker.py'),'--socket',str(sock)],stdout=out,stderr=subprocess.STDOUT)
                self.sockets[key]=sock;self.processes[key]=proc
        deadline=time.monotonic()+12
        while not all(s.exists() for s in self.sockets.values()):
            assert all(p.poll() is None for p in self.processes.values()),'worker exited during startup'
            assert time.monotonic()<deadline,'worker startup timeout';time.sleep(.03)
        self.owners={f'{s}:{b}':p.pid for (s,b),p in self.processes.items()};self.set_available('all')
    def set_available(self,which):
        assert which in ['all','cpu','gstreamer']
        for (stage,backend),p in self.processes.items():
            assert p.poll() is None
            os.kill(p.pid,signal.SIGCONT if which=='all' or backend==which else signal.SIGSTOP)
        self.availability=which
    def call(self,req):
        key=(req['stage'],req['backend']);assert key in self.sockets
        request=dict(req,deadline_unix=time.time()+2.0);sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(2.2)
        started=time.monotonic()
        try:
            sock.connect(str(self.sockets[key]));sock.sendall((json.dumps(request)+'\n').encode());stream=sock.makefile('rb');line=stream.readline(1048577)
            assert line,'worker closed without result';result=json.loads(line);assert result['request_id']==req['request_id']
        except (TimeoutError,socket.timeout):result=dict(ok=False,error='actual_worker_RPC_timeout',request_id=req['request_id'])
        except BlockingIOError as exc:result=dict(ok=False,error='actual_worker_endpoint_backpressure',errno=exc.errno,request_id=req['request_id'],executed=False)
        finally:sock.close()
        return dict(request=request,result=result,host_elapsed_seconds=time.monotonic()-started)
    def close(self):
        for p in self.processes.values():
            if p.poll() is None:os.kill(p.pid,signal.SIGCONT)
        for key,p in self.processes.items():
            if p.poll() is None:
                try:
                    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(.5);sock.connect(str(self.sockets[key]));sock.sendall(b'{"op":"stop"}\n');sock.close();p.wait(timeout=3)
                except Exception:
                    p.terminate()
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            self.sockets[key].unlink(missing_ok=True)
        for f in self.logs:f.close()
