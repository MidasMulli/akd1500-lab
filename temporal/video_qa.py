"""Decode both temporal films; verify cadence, unique frames and fast start."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess
def main(folder):
    rows=[]
    for p in sorted(folder.glob('akd-temporal-context-*.mp4')):
        r=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(p)]))['streams'][0]
        raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(p),'-an','-f','framemd5','-'],text=True);frames=[x.rsplit(',',1)[-1].strip() for x in raw.splitlines() if x and not x.startswith('#')];assert len(frames)==len(set(frames))==1440 and r['r_frame_rate']=='60/1' and float(r['duration'])==24
        b=p.read_bytes();at=0;boxes=[]
        while at+8<=len(b):
            n,tag=struct.unpack_from('>I4s',b,at)
            if n==1:n=struct.unpack_from('>Q',b,at+8)[0]
            if n==0:n=len(b)-at
            assert n>=8;boxes.append((tag.decode('ascii'),at,n));at+=n
        assert at==len(b) and next(x[1] for x in boxes if x[0]=='moov')<next(x[1] for x in boxes if x[0]=='mdat');assert [r['width'],r['height']] in [[3840,2160],[1920,1080]]
        rows.append(dict(file=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),**r,decoded_frames=len(frames),unique_frames=len(set(frames)),fast_start=True))
    assert len(rows)==2;result=dict(status='PASS',outputs=rows);(folder/'VIDEO_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
