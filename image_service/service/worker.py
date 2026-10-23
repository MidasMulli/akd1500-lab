"""Owned image workers. No AKD access, no routing/fault-schedule knowledge."""
from pathlib import Path
import argparse,hashlib,json,os,socket,subprocess,time,traceback
import cv2
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def operation(req):
    src=Path(req['input']).resolve();out=Path(req['output']).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    backend=req['backend'];stage=req['stage'];assert backend in ['cpu','gstreamer'] and stage in ['decode','export']
    commands=[]
    if stage=='decode':
        width,height=req['width'],req['height']
        if backend=='cpu':
            rgb=cv2.imread(str(src),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION);assert rgb is not None
            rgba=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGBA)
        else:
            raw=out.with_suffix('.rgba')
            cmd=['gst-launch-1.0','-q','-e','filesrc','location='+str(src),'!','jpegparse','!','jpegdec','!',
                 'videoconvert','!','video/x-raw,format=RGBA','!','filesink','location='+str(raw)]
            commands.append(cmd);pr=subprocess.run(cmd,capture_output=True,text=True,timeout=8)
            if pr.returncode:raise RuntimeError('gstreamer decode: '+pr.stderr[-3000:])
            assert raw.stat().st_size==width*height*4
            rgba=np.frombuffer(raw.read_bytes(),np.uint8).reshape(height,width,4).copy()
    else:
        rgba=np.load(src,allow_pickle=False);height,width,_=rgba.shape
        if backend=='cpu':rgba=cv2.resize(rgba,(320,180),interpolation=cv2.INTER_LINEAR)
        else:
            rawin=out.with_suffix('.input.rgba');rawout=out.with_suffix('.output.rgba');rawin.write_bytes(rgba.tobytes())
            cmd=['gst-launch-1.0','-q','-e','filesrc','location='+str(rawin),'!','rawvideoparse','format=rgba',
                 'width='+str(width),'height='+str(height),'framerate=1/1','!','nvvidconv','interpolation-method=1','!',
                 'video/x-raw,format=RGBA,width=320,height=180','!','filesink','location='+str(rawout)]
            commands.append(cmd);pr=subprocess.run(cmd,capture_output=True,text=True,timeout=8)
            if pr.returncode:raise RuntimeError('gstreamer resize: '+pr.stderr[-3000:])
            assert rawout.stat().st_size==320*180*4
            rgba=np.frombuffer(rawout.read_bytes(),np.uint8).reshape(180,320,4).copy()
    assert rgba.dtype==np.uint8 and rgba.ndim==3 and rgba.shape[2]==4
    with out.open('wb') as f:np.save(f,rgba,allow_pickle=False)
    png=None
    if stage=='export':
        png=out.with_suffix('.png');ok=cv2.imwrite(str(png),cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA));assert ok
    return dict(ok=True,stage=stage,backend=backend,input_sha256=sha(src),output_sha256=sha(out),
                output=str(out),png=str(png) if png else None,png_sha256=sha(png) if png else None,
                shape=list(rgba.shape),commands=commands)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--socket',required=True);args=ap.parse_args();path=Path(args.socket)
    assert not path.exists();sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.bind(str(path));sock.listen(8)
    try:
        while True:
            conn,_=sock.accept()
            with conn:
                stream=conn.makefile('rwb');line=stream.readline(65537)
                if not line:continue
                try:
                    req=json.loads(line);start=time.monotonic()
                    if req.get('op')=='stop':break
                    if time.time()>req['deadline_unix']:result=dict(ok=False,error='expired_before_execution',executed=False)
                    else:
                        try:result=operation(req);result['executed']=True
                        except Exception as exc:result=dict(ok=False,error=str(exc),traceback=traceback.format_exc(),executed=True)
                    result['worker_pid']=os.getpid();result['request_id']=req['request_id'];result['elapsed_seconds']=time.monotonic()-start
                    try:stream.write((json.dumps(result)+'\n').encode());stream.flush()
                    except (BrokenPipeError,ConnectionResetError):pass
                except (json.JSONDecodeError,KeyError):pass
    finally:sock.close();path.unlink(missing_ok=True)
if __name__=='__main__':main()
