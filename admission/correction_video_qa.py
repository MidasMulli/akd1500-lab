"""Full software/hardware decode, 60-fps cadence and fast-start checks."""
from pathlib import Path
import argparse,json,struct,subprocess
from admission.run import sha
def main(folder,pattern='native-memory-correction-*.mp4'):
    rows=[]
    for p in sorted(folder.glob(pattern)):
        r=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration,profile,level,pix_fmt,has_b_frames','-of','json',str(p)]))['streams'][0]
        raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(p),'-an','-f','framemd5','-'],text=True)
        frames=[x.rsplit(',',1)[-1].strip() for x in raw.splitlines() if x and not x.startswith('#')]
        assert len(frames)==len(set(frames))==1800 and r['r_frame_rate']=='60/1' and float(r['duration'])==30
        assert r['profile']=='Main' and r['pix_fmt']=='yuv420p' and r['has_b_frames']==0
        assert [r['width'],r['height'],r['level']] in [[3840,2160,52],[1920,1080,42]]
        result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-hwaccel','videotoolbox','-i',str(p),'-an','-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0 and not result.stderr,result.stderr
        b=p.read_bytes();at=0;boxes=[]
        while at+8<=len(b):
            n,tag=struct.unpack_from('>I4s',b,at)
            if n==1:n=struct.unpack_from('>Q',b,at+8)[0]
            if n==0:n=len(b)-at
            assert n>=8;boxes.append((tag.decode('ascii'),at,n));at+=n
        assert at==len(b) and next(q[1] for q in boxes if q[0]=='moov')<next(q[1] for q in boxes if q[0]=='mdat')
        rows.append(dict(file=p.name,bytes=len(b),sha256=sha(p),**r,decoded_frames=len(frames),unique_frames=len(set(frames)),software_decode='PASS',apple_videotoolbox_decode='PASS',fast_start=True))
    assert len(rows)==2
    result=dict(status='PASS',outputs=rows);(folder/'VIDEO_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--pattern',default='native-memory-correction-*.mp4');a=p.parse_args();main(a.folder,a.pattern)
