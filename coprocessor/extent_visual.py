"""Recorded larger-graph replay using the existing schematic renderer."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import ImageDraw
from coprocessor.extent_audit import BASE,audit,masks,sha
from coprocessor.resident_visual import ResidentScene
from coprocessor.neuron_visual import text,BG,INK,MUTED,CYAN

def source_data():
    checked=audit();meta=json.loads(BASE.with_suffix('.json').read_text());cell='s64_s107501';f=meta['fixtures'][cell]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        fg=lambda k:z[f['arrays'][k]['key']];initial=fg('ram');features=fg('features').reshape(64,512)
        get=lambda arm,k:z[meta['runs'][f'native_{cell}_a{arm}_n64_r0']['arrays'][k]['key']]
        states=get(1,'snapshots').reshape(64,len(initial));frozen=get(0,'snapshots').reshape(64,len(initial))
        before=np.concatenate([initial[None],states[:-1]]);bm,am=masks(before,512),masks(states,512);fm=masks(frozen,512)
        scores=get(1,'scores').reshape(64,128);fscores=get(0,'scores').reshape(64,128)
        shown=[8,12,16,20,24,28,32,36];units=[int(np.flatnonzero(np.any(bm[i]!=am[i],axis=1))[0]) for i in shown]
        assert len(set(units))==8
        events=[]
        for i in shown+list(range(40,48)):
            delta=am[i].astype(int)-bm[i].astype(int);edits=np.flatnonzero(np.any(delta!=0,axis=1)).tolist()
            assert np.array_equal(frozen[i],initial)
            events.append(dict(index=i,query=i>=40,input=features[i].tolist(),units=units,
                before=bm[i,units].tolist(),after=am[i,units].tolist(),frozen=fm[i,units].tolist(),
                response=scores[i].tolist(),frozen_response=fscores[i].tolist(),edited_records=edits,
                added_connections=int((delta==1).sum()),removed_connections=int((delta==-1).sum())))
    return dict(seed=107501,side=64,features=512,units=units,events=events,
                archive_sha256=meta['archive_sha256'],native_mapping=f['metadata']['mapping'])

class ExtentScene(ResidentScene):
    def __init__(self,data):
        super().__init__(data);d=ImageDraw.Draw(self.base)
        d.rectangle((100,45,3740,255),fill=BG)
        text(d,(115,65),'A larger stream. The memory still learns on-chip.',82,INK,True)
        text(d,(120,190),'64×64 RGB → 512 feature bits → native connection changes → read-only queries',39,MUTED)
        d.rectangle((150,690,2550,742),fill=(11,20,31))
        text(d,(160,695),'512 derived feature bits · 8 sampled records · 32 stored connections each',31,MUTED)
        d.rectangle((1470,443,2095,499),fill=(13,24,36))
        text(d,(1475,449),'8 binary channels × 8×8',27,MUTED)
        d.rectangle((110,2082,3730,2150),fill=BG)
        text(d,(120,2094),'3 fresh initializations · 5 exact CPU controls · CPU remains faster · power rig not yet assembled',29,MUTED)
        j=np.arange(512);yy=1-2*(j+.5)/512;rr=np.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5))
        self.features=np.column_stack((-1.15+.27*rr*np.cos(theta),yy*1.12,.85*rr*np.sin(theta)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    data=source_data();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');scene=ExtentScene(data)
    for name,t in [('teaching',.5),('edit',1.5),('later-edit',9.5),('recall',18.5)]:scene.frame(t).save(a.out/(name+'.png'))
    if a.video:
        output=a.out/'larger-native-stream-4k60.mp4'
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','26M','-profile:v','high','-bsf:v','h264_metadata=level=5.2','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(output)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1440):
                proc.stdin.write(scene.frame(i/60).tobytes())
                if i%240==0:print(json.dumps(dict(frame=i,total=1440,elapsed=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(output),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'larger-native-stream-1080p60.mp4')],check=True)
    binding=dict(status='RECORDED_MEASUREMENTS_SCHEMATIC_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(Path(__file__)),helper_renderer_sha256=sha(Path(__file__).with_name('resident_visual.py')),
        sampled_records=data['units'],shown_teaching_events=[9,13,17,21,25,29,33,37],shown_query_events=list(range(41,49)),
        measured=['Returned pre-update learner potentials','Complete learner RAM before/after','CNP config readbacks'],
        derived=['512 intermediate feature bits from independent integer convolution'],
        schematic=['Positions','Links as geometry','Camera','Expanded playback timing'],internal_electrical_spike_times_observed=False,
        outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.mp4','.png']})
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
