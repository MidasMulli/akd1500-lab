"""Recorded learning plus measured host costs; no invented electrical telemetry."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,tarfile,time
import numpy as np
from PIL import ImageDraw
from coprocessor.extent_visual import ExtentScene
from coprocessor.neuron_visual import text,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
from coprocessor.extent_audit import masks
from coprocessor.service_archive import BASE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert sha(str(BASE)+'.tar.gz')==meta['archive_sha256']
    with tarfile.open(str(BASE)+'.tar.gz','r:gz') as tar:
        def raw(path):
            value=tar.extractfile(path).read();assert hashlib.sha256(value).hexdigest()==meta['files'][path];return value
        def arr(path,dtype):return np.frombuffer(raw(path),dtype).copy()
        f='fixtures/s64_s107501_potentials/';initial=arr(f+'cell.ram','<u4');features=arr(f+'cell.features','u1').reshape(64,512)
        n='raw/native_s64_s107501_potentials_a1_n64_r1_w100_b1_p1_f1/';frozen='raw/native_s64_s107501_potentials_a0_n64_r1_w100_b1_p1_f1/'
        states=arr(n+'result.snapshots','<u4').reshape(64,-1);fr=arr(frozen+'result.snapshots','<u4').reshape(64,-1);before=np.concatenate([initial[None],states[:-1]])
        bm,am,fm=masks(before,512),masks(states,512),masks(fr,512);scores=arr(n+'result.scores','<i4').reshape(64,128);fscores=arr(frozen+'result.scores','<i4').reshape(64,128)
        shown=[8,12,16,20,24,28,32,36];units=[int(np.flatnonzero(np.any(bm[i]!=am[i],axis=1))[0]) for i in shown];assert len(set(units))==8
        events=[]
        for i in shown+list(range(40,48)):
            delta=am[i].astype(int)-bm[i].astype(int);assert np.array_equal(fr[i],initial)
            events.append(dict(index=i,query=i>=40,input=features[i].tolist(),units=units,before=bm[i,units].tolist(),after=am[i,units].tolist(),frozen=fm[i,units].tolist(),response=scores[i].tolist(),frozen_response=fscores[i].tolist(),edited_records=np.flatnonzero(np.any(delta!=0,axis=1)).tolist(),added_connections=int((delta==1).sum()),removed_connections=int((delta==-1).sum())))
        costs=[]
        for key,label,color in [('cpu-blas4','CPU · OpenBLAS4',(167,177,226)),('native-spin','AKD · spin',AMBER),('native-sleep','AKD · sleep100µs',GREEN)]:
            folder='raw/'+('cpu-blas4_s64_s107501_potentials_a1_n64_r1_w0_b1_p1_f1' if key=='cpu-blas4' else 'native_s64_s107501_potentials_a1_n64_r1_w'+('0' if key=='native-spin' else '100')+'_b1_p1_f1')+'/'
            costs.append(dict(key=key,label=label,color=color,wall=(arr(folder+'result.latency_ns','<i8')/1e6).tolist(),cpu=(arr(folder+'result.caller_ns','<i8')/1e6).tolist()))
        forecasts=[]
        for name,label,color in [('none','Model only',MUTED),('cpu-blas1','+ CPU learner',(151,177,225)),('native','+ AKD sleep100µs',GREEN)]:
            folder=f'window_raw/b3_{name}_s107501_a1_n64_w{100 if name=="native" else 0}_r0/'
            owner=json.loads(raw(folder+'RECEIPT.json'));timing=np.load(io.BytesIO(raw(folder+'foreground-timing.npy')),allow_pickle=False)
            ends=((timing[:,1]-owner['start_ns'])/1e9);forecasts.append(dict(label=label,color=color,completion_seconds=ends[ends<=3.2].tolist()))
        mapping=json.loads(raw(f+'BUILD.json'))['mapping']
    return dict(seed=107501,side=64,features=512,units=units,events=events,costs=costs,forecasts=forecasts,native_mapping=mapping,archive_sha256=meta['archive_sha256'],source_run=n,scope='Full-RAM learning replay and cost comparison use separate matched confirmation runs. Foreground counts use separate3.2s common-clock block0 trials. Playback/camera/geometry are schematic; no electrical spike times measured.')

class ServiceScene(ExtentScene):
    def __init__(self,data):
        super().__init__(data);d=ImageDraw.Draw(self.base)
        d.rectangle((100,45,3740,255),fill=BG)
        text(d,(115,65),'Learning stays on-chip. The host polls less.',76,INK,True)
        text(d,(120,190),'Exact native updates retained · far less caller CPU · CPU still returns answers faster',38,MUTED)
        d.rectangle((110,1930,3740,2160),fill=BG)
        text(d,(120,1932),'Recorded: returned match counts, full learning-memory boundaries and per-call wall/process CPU costs.',30,MUTED)
        text(d,(120,1988),'Separate co-run: frozen Nano T202 forecasts + 20Hz learning stream. No ≥5% throughput win over both CPU controls.',29,MUTED)
        text(d,(120,2044),'Schematic geometry, camera and expanded playback time. Feature bits derived; no electrical spike timing or energy measured.',28,MUTED)
        text(d,(120,2096),'80 physical sessions · 4,672 forwards · three initializations · five exact CPU implementations',29,MUTED)
    def frame(self,t):
        im=super().frame(t);d=ImageDraw.Draw(im);d.rounded_rectangle((2630,590,3730,1770),radius=32,fill=(13,22,32),outline=(39,58,76),width=2)
        text(d,(2690,625),'MEASURED HOST COST',36,CYAN,True)
        text(d,(2690,688),'Same event · separate matched runs',28,MUTED)
        j=min(7,int(t/2)) if t<16 else 8+min(7,int(t-16));i=self.data['events'][j]['index']
        for k,row in enumerate(self.data['costs']):
            y=755+k*154;cpu=row['cpu'][i];wall=row['wall'][i]
            text(d,(2690,y),row['label'],30,row['color'],True)
            text(d,(3270,y+3),f'{wall:.2f} ms answer',26,MUTED)
            d.rectangle((2690,y+57,3620,y+77),fill=(27,41,54));d.rectangle((2690,y+57,2690+min(930,930*cpu/10),y+77),fill=row['color'])
            text(d,(2690,y+91),f'{cpu:.2f} ms process CPU consumed',27,INK)
        text(d,(2690,1245),'0',24,MUTED);text(d,(3500,1245),'10 ms',24,MUTED)
        d.line((2690,1300,3650,1300),fill=DIM,width=2)
        text(d,(2690,1330),'SEPARATE NANO CO-RUN',31,CYAN,True)
        text(d,(2690,1380),'3.2s recorded window · time-expanded',25,MUTED)
        elapsed=min(3.2,t/24*3.2)
        for k,row in enumerate(self.data['forecasts']):
            y=1440+k*88;count=int(np.searchsorted(row['completion_seconds'],elapsed,side='right'))
            text(d,(2690,y),row['label'],26,row['color'])
            d.rectangle((2690,y+38,3500,y+55),fill=(27,41,54));d.rectangle((2690,y+38,2690+810*count/140,y+55),fill=row['color']);text(d,(3540,y+22),str(count),33,INK,True)
        text(d,(2690,1720),'Completed forecasts · effectively tied',26,MUTED)
        d.rectangle((2630,1790,3740,1875),fill=BG)
        text(d,(2690,1820),'CPU cost + foreground completions',27,MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=source();(a.out/'recorded-source.json').write_text(json.dumps(data,separators=(',',':'))+'\n');scene=ServiceScene(data)
    for name,t in [('teaching',.5),('native-edit',1.5),('query',18.5),('final',23.999)]:scene.frame(t).save(a.out/(name+'.png'))
    if a.video:
        path=a.out/'native-learning-host-sleep-4k60.mp4';cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','24M','-profile:v','high','-bsf:v','h264_metadata=level=5.2','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1440):
                proc.stdin.write(scene.frame(i/60).tobytes())
                if i%240==0:print(json.dumps(dict(frame=i,total=1440,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(path),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'native-learning-host-sleep-1080p60.mp4')],check=True)
    binding=dict(status='RECORDED_NUMERIC_REPLAY',archive_sha256=data['archive_sha256'],source_run=data['source_run'],recorded_source_sha256=sha(a.out/'recorded-source.json'),renderer_sha256=sha(__file__),scope=data['scope'],measured=['Full128 returned potentials','Full learning RAM boundaries','Per-call process CPU and wall clocks','Foreground forecast completion clocks'],derived=['Intermediate512 feature bits','Connection additions/removals from measured RAM deltas'],schematic=['Geometry','Camera','Time expansion'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']})
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))

if __name__=='__main__':main()
