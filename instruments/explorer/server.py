"""Private explorer HTTP server. Replay reads never initiate hardware operations."""
from __future__ import annotations
import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import os
import shlex
import re
import subprocess
import threading
import time
from urllib.parse import urlparse,parse_qs
from instruments.explorer.tablet import compact
from instruments.explorer.live import validate_action

HERE=Path(__file__).resolve().parent


class Bridge:
    def __init__(self,enable_native=False):
        self.enable_native=enable_native;self.proc=None;self.condition=threading.Condition()
        self.serial=threading.Lock();self.generation=0
        self.state=dict(status='idle',events=[],initial=None,mode=None,arm=None,error=None)

    def snapshot(self):
        with self.condition:return json.loads(json.dumps(self.state))

    def read(self,proc,generation):
        for line in proc.stdout:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            with self.condition:
                if generation!=self.generation:return
                typ=r.get('type')
                if typ=='started':self.state.update(r,status='bootstrapping',events=[])
                elif typ=='event':
                    self.state['events'].append(r['event']);self.state['received_at']=time.time()
                elif typ=='ready':self.state.update(status='ready',completed=r['events'])
                elif typ=='stopped':self.state.update(status='stopped',receipt=r.get('receipt'))
                elif typ=='error':self.state.update(status='failed',error=r.get('error'))
                self.condition.notify_all()
        with self.condition:
            if generation==self.generation and self.state['status'] not in ['stopped','failed']:
                self.state.update(status='disconnected',error='Worker connection closed; inspect its receipt before restarting.')
            self.condition.notify_all()

    def command(self,a):
        a=validate_action(a)
        if a['action']=='start' and a['mode']!='native':
            raise ValueError('Browser sessions use the native chip; CPU qualification is a separate local operation')
        if a['action']=='start' and a['mode']=='native' and not self.enable_native:
            raise ValueError('Native controls are not admitted on this server')
        if not self.serial.acquire(blocking=False):raise ValueError('An operation is already running')
        try:
            if a['action']=='start':
                if self.proc and self.proc.poll() is None:raise ValueError('Stop the current worker first')
                ssh_host=os.environ.get('AKD_SSH_HOST','');remote_root=os.environ.get('AKD_REMOTE_ROOT','');remote_python=os.environ.get('AKD_REMOTE_PYTHON','')
                if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',ssh_host) or not remote_root.startswith('/') or not remote_python.startswith('/'):
                    raise ValueError('Configure AKD_SSH_HOST, AKD_REMOTE_ROOT and AKD_REMOTE_PYTHON before a qualified live session')
                with self.condition:
                    self.generation+=1;self.state=dict(status='starting',events=[],initial=None,mode=a['mode'],arm=a['arm'],error=None)
                with (HERE/'generated/ssh-worker.stderr').open('a') as worker_log:
                    self.proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','ServerAliveInterval=10',
                    '-o','ServerAliveCountMax=2',ssh_host,
                    'cd '+shlex.quote(remote_root)+' && OPENBLAS_NUM_THREADS=1 '+shlex.quote(remote_python)+' -u -m instruments.explorer.live'],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=worker_log,text=True,bufsize=1)
                threading.Thread(target=self.read,args=(self.proc,self.generation),daemon=True).start()
            if not self.proc or self.proc.poll() is not None:
                if a['action']=='stop':return self.snapshot()
                raise ValueError('Start a fresh session first')
            with self.condition:self.state['status']='busy'
            self.proc.stdin.write(json.dumps(a)+'\n');self.proc.stdin.flush()
            end=time.monotonic()+30
            with self.condition:
                while self.state['status'] not in ['ready','stopped','failed','disconnected']:
                    remaining=end-time.monotonic()
                    if remaining<=0:raise TimeoutError('Operation is still running; inspect live status before another action')
                    self.condition.wait(remaining)
            if a['action']=='stop':
                # EOF closes only our dedicated SSH worker after native cleanup.
                self.proc.stdin.close()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.terminate()
            return self.snapshot()
        finally:self.serial.release()

    def close(self):
        try:self.command(dict(action='stop'))
        except Exception:pass


def make_handler(bridge,replay_path,port,public_host):
    csrf=secrets.token_urlsafe(32)
    raw=Path(replay_path).read_bytes();receipt=json.loads(Path(replay_path).with_suffix('.receipt.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=receipt['data_sha256']:raise ValueError('export hash changed')
    compressed=gzip.compress(raw);tablet_data=json.loads(raw)
    readonly_profile=tablet_data.get('live_compatible') is False
    if (readonly_profile or len(tablet_data['banks'])!=3) and bridge.enable_native:raise ValueError('This recorded profile has no qualified native actor; omit --enable-native')
    hosts={f'{host}:{port}' for host in ['localhost','127.0.0.1',public_host]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,body,kind='application/json',encoding=None):
            if not isinstance(body,bytes):body=json.dumps(body).encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if encoding:self.send_header('Content-Encoding',encoding)
            self.end_headers();self.wfile.write(body)
        def valid_host(self):return self.headers.get('Host') in hosts
        def do_GET(self):
            if not self.valid_host():return self.send(421,dict(error='Host rejected'))
            parsed=urlparse(self.path);path=parsed.path
            if path in ('/api/tab5','/api/tab5/v2'):
                if readonly_profile:return self.send(400,dict(error='This RGB profile has no qualified tablet projection'))
                try:
                    snap=bridge.snapshot();busy=(HERE/'generated/RESEARCH_BUSY.json').exists()
                    return self.send(200,compact(tablet_data,snap,parse_qs(parsed.query,keep_blank_values=True),received_at=snap.get('received_at'),research_busy=busy,schema=2 if path.endswith('/v2') else 1))
                except ValueError as e:return self.send(400,dict(error=str(e)))
            if path=='/api/replay':return self.send(200,compressed,'application/json','gzip')
            if path=='/api/status':return self.send(200,dict(csrf=csrf,native_enabled=bridge.enable_native,export=receipt,live=bridge.snapshot()))
            if path=='/api/live':
                if readonly_profile:return self.send(409,dict(error='This recorded RGB profile has no qualified live actor'))
                return self.send(200,bridge.snapshot())
            name='index.html' if path=='/' else path.lstrip('/')
            file=(HERE/'web'/name).resolve()
            if not file.is_relative_to((HERE/'web').resolve()) or not file.is_file():return self.send(404,dict(error='Not found'))
            kinds={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}
            if file.suffix not in kinds:return self.send(404,dict(error='Not found'))
            return self.send(200,file.read_bytes(),kinds[file.suffix])
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.valid_host() or (origin and origin not in {'http://'+h for h in hosts}) or self.headers.get('X-Explorer-Token')!=csrf:
                return self.send(403,dict(error='Same-origin operation token required'))
            if self.path!='/api/live':return self.send(404,dict(error='Not found'))
            if readonly_profile:return self.send(409,dict(error='Recorded RGB inspection cannot invoke the three-bank actor'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Bounded request required')
                a=json.loads(self.rfile.read(length));validate_action(a)
                if a.get('action')=='start' and (HERE/'generated/RESEARCH_BUSY.json').exists():raise ValueError('Overnight research owns the chip; read-only inspection remains available')
                self.send(200,bridge.command(a))
            except (ValueError,TimeoutError,RuntimeError,BrokenPipeError) as e:self.send(409,dict(error=str(e)))
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8892);p.add_argument('--bind',default='127.0.0.1')
    p.add_argument('--public-host',default='localhost');p.add_argument('--enable-native',action='store_true')
    p.add_argument('--replay',type=Path,default=HERE/'generated/replay.json')
    a=p.parse_args();(HERE/'generated').mkdir(exist_ok=True)
    if a.enable_native:
        gate=json.loads((HERE/'generated/NATIVE_QUALIFICATION.json').read_text())
        if gate.get('status')!='NATIVE_QUERY_TEACH_PASS':raise ValueError('Native qualification gate is closed')
        for relative,digest in gate['source_hashes'].items():
            if hashlib.sha256((HERE/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Qualified source changed')
    b=Bridge(a.enable_native);server=ThreadingHTTPServer((a.bind,a.port),make_handler(b,a.replay,a.port,a.public_host))
    print(json.dumps(dict(status='EXPLORER_SERVING',bind=a.bind,port=a.port,native_enabled=a.enable_native)),flush=True)
    try:server.serve_forever()
    finally:b.close();server.server_close()


if __name__=='__main__':main()
