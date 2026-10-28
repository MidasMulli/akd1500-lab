"""Read-only local/private replay server with bounded MP4 byte-range support."""
import argparse,functools,http.server,mimetypes,re,signal,threading
from pathlib import Path

class ReplayHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining=None
        root=Path(self.directory).resolve();p=Path(self.translate_path(self.path)).resolve()
        if p.is_dir():p=p/'index.html'
        if not p.is_relative_to(root):self.send_error(404);return
        rel=p.relative_to(root)
        allowed=(len(rel.parts)==1 and (p.name in {'index.html','film.js','data.json','DATA_QA.json','VIDEO_DELIVERY_QA.json'} or p.suffix=='.mp4')) or (rel.parts[0]in {'img','vendor'} and p.suffix in {'.png','.js'})
        if not allowed or not p.is_file():self.send_error(404);return
        f=p.open('rb');n=p.stat().st_size;start,end=0,n-1;partial=False
        header=self.headers.get('Range')
        if header:
            m=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
            valid=bool(m and (m[1]or m[2]) and n)
            if valid:
                if m[1]:start=int(m[1]);end=min(int(m[2]),n-1)if m[2]else n-1
                else:
                    length=int(m[2]);valid=length>0;start=max(0,n-length)
                valid=valid and 0<=start<n and start<=end
            if not valid:
                f.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{n}');self.send_header('Content-Length','0');self.end_headers();return
            partial=True
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',mimetypes.guess_type(str(p))[0]or'application/octet-stream')
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{n}')
        self.end_headers();f.seek(start);self.remaining=end-start+1;return f
    def copyfile(self,source,outputfile):
        while self.remaining:
            b=source.read(min(256*1024,self.remaining))
            if not b:break
            try:outputfile.write(b)
            except (BrokenPipeError,ConnectionResetError):break
            self.remaining-=len(b)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--port',type=int,default=8904);p.add_argument('--bind',action='append',default=[]);a=p.parse_args()
    servers=[http.server.ThreadingHTTPServer((h,a.port),functools.partial(ReplayHandler,directory=str(a.directory)))for h in a.bind or ['127.0.0.1']]
    stopped=threading.Event()
    for s in servers:threading.Thread(target=s.serve_forever,daemon=True).start()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopped.set())
    stopped.wait()
    for s in servers:s.shutdown();s.server_close()
