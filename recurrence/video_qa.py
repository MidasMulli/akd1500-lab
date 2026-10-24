"""Decode both rendered films and verify frame count, cadence and fast-start."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess
def main(folder):
    rows=[]
    for f in sorted(folder.glob('akd-native-feedback-*.mp4')):
        r=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(f)]))['streams'][0]
        decoded=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(f),'-an','-f','framemd5','-'],text=True)
        hashes=[x.rsplit(',',1)[-1].strip() for x in decoded.splitlines() if x and not x.startswith('#')]
        data=f.read_bytes();at=0;boxes=[]
        while at+8<=len(data):
            n,tag=struct.unpack_from('>I4s',data,at)
            if n==1:n=struct.unpack_from('>Q',data,at+8)[0]
            if n==0:n=len(data)-at
            assert n>=8;boxes.append((tag.decode('ascii'),at,n));at+=n
        assert at==len(data) and len(hashes)==len(set(hashes))==1440 and r['r_frame_rate']=='60/1' and float(r['duration'])==24
        assert [r['width'],r['height']] in [[3840,2160],[1920,1080]]
        assert next(x[1] for x in boxes if x[0]=='moov')<next(x[1] for x in boxes if x[0]=='mdat')
        rows.append(dict(file=f.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**r,decoded_frames=len(hashes),unique_decoded_frames=len(set(hashes)),fast_start=True))
    assert len(rows)==2
    result=dict(status='PASS',outputs=rows);(folder/'VIDEO_QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
