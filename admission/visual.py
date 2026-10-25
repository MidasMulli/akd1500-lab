"""Measured three-bank replay. Camera/placement/flow timing are schematic."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from admission.archive import BASE,audit,sha
from admission.learned import simulate,DIMENSIONS,P0,P1,SEEDS,KEYS,RAW,OUTCOME
from admission.run import masks
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H,FPS,DURATION=3840,2160,60,32

def source():
    checked=audit();m=json.loads(BASE.with_suffix('.json').read_text());fam=m['families']['learned'];seed=SEEDS[0];data=dict(archive_sha256=checked['archive_sha256'],summary={k:checked[k] for k in ['primary','cpu','retained_original_record_ids']},arms={})
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for arm in ['native','open','frozen','manager_frozen','severed']:
            r=fam['runs'][f'native_s{seed}_{arm}_heldout'];f=fam['fixtures'][f's{seed}_{arm}'];get=lambda k:z[r['arrays'][k]['key']]
            if arm!='native':data['arms'][arm]=dict(outputs=get('scores').reshape(-1,128).tolist());continue
            rams=[z[f['arrays'][f'bank{k}.ram']['key']] for k in range(3)];a=simulate(rams,'heldout',arm)
            bank=[]
            for k,(n,u,c) in enumerate(DIMENSIONS):
                plastic=[0] if k==0 else P0 if k==1 else P1;v=masks(get(f'states{k}').reshape(-1,len(rams[k])),n,u)[:,plastic];initial=masks(rams[k],n,u)[plastic];before=np.concatenate([initial[None],v[:-1]]);adds=((v==1)&(before==0)).sum((1,2));removed=((v==0)&(before==1)).sum((1,2))
                bank.append(dict(initial=initial.tolist(),connections=v.tolist(),adds=adds.tolist(),removed=removed.tolist(),touched=(v!=initial).any(-1).sum(-1).tolist()))
            data['arms'][arm]=dict(events=a['events'],inputs=get('inputs').reshape(-1,128).tolist(),outputs=get('scores').reshape(-1,128).tolist(),banks=bank,features=[a[f'features{k}'].tolist() for k in range(3)])
    data['scope']='Recorded native binary responses and all three learning RAMs. One plastic recency record plus two eight-record consequence banks; fixed helper records omitted from connection clouds and explicitly counted. Fixed encoding and repeat-admission rule are designed, manager recent-key state is learned. Native output permission directly feeds the teacher input; host-paced uniform phase clocks and actual outcomes. Intermediate fixed features derived, camera/geometry/flow timing schematic, selected slowed events, no electrical spike or physical-die trace. CPU repeat-admission also passes; no timing/energy advantage.'
    return data

def prediction(y,start=64):
    b=np.asarray(y[start:start+8]).reshape(2,4).any(1)
    return '0 + 1' if b.all() else '0' if b[0] else '1' if b[1] else '—'

class Film:
    def __init__(self,data):
        self.data=data;self.a=data['arms']['native'];self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(110,65),'Learning controls learning.',94,INK,True)
        text(d,(115,205),'Native synaptic state decides what enters a protected memory bank.',42,MUTED)
        text(d,(2960,90),'AKD1500 · RECORDED HARDWARE',29,CYAN,True)
        text(d,(2960,145),'Native connection RAM + returned responses',25,MUTED)
        d.line((115,285,3725,285),fill=DIM,width=2)
        for box in [(115,330,3725,465),(115,510,2495,1775),(2550,510,3725,1775)]:d.rounded_rectangle(box,radius=26,fill=(11,20,31),outline=DIM,width=2)
        text(d,(160,545),'SEVEN FNP3s · THREE RESIDENT PLASTIC GROUPS',29,CYAN,True)
        text(d,(160,600),'Fixed encoders / native gate / fixed relays connect these groups.',27,MUTED)
        text(d,(160,645),'Responses precede updates; colored edges overlay the resulting stored edits.',25,MUTED)
        for xx,label,sub in [(180,'SYNAPTIC MANAGER','FNP3 [2,1,0] · 1 plastic record'),(970,'ALWAYS-UPDATE BANK','FNP3 [2,3,0] · 8 plastic records'),(1760,'PROTECTED BANK','FNP3 [3,2,0] · 8 plastic records')]:
            text(d,(xx,700),label,30,CYAN,True);text(d,(xx,750),sub,25,MUTED)
        for xx,n in [(180,31),(970,40),(1760,24)]:text(d,(xx,791),f'+ {n} fixed forwarding helpers (not drawn)',23,MUTED)
        p=data['summary']['primary'];cpu=data['summary']['cpu'];nv=p['native'];ov=p['open'];fv=p['frozen'];cv=cpu['repeat_lru']
        controls=f'{ov["heldout_correct"]} / 480' if ov['heldout_correct']==fv['heldout_correct'] else f'{ov["heldout_correct"]} vs {fv["heldout_correct"]} / 480'
        for xx,label,value,col in [(115,'USEFUL MEMORY REVISIT',f'{nv["final_probe_correct"]} / 120',GREEN),(1040,'HELD-OUT ORDER',f'{nv["heldout_correct"]} / 480',GREEN),(1965,'ALWAYS-OPEN / FROZEN',controls,AMBER),(2890,'CPU REPEAT-ADMISSION',f'{cv["heldout_correct"]} / 480',CYAN)]:
            d.rounded_rectangle((xx,1810,xx+835,1935),radius=18,fill=(14,25,36),outline=DIM,width=2);text(d,(xx+25,1827),label,24,MUTED,True);text(d,(xx+25,1864),value,43,col,True)
        for yy,line in [(1970,'Five initializations · sealed synthetic 12-key stream · 8 records per bank · repeat rule designed; recent-key state learned.'),(2016,'Host sets fixed per-bank phase clocks and sends raw keys / outcomes; no data-dependent bank choice or weight repair.'),(2062,'Recorded outputs + connection RAM. Intermediate features derived. Geometry, camera and flow schematic; not electrical spikes.'),(2108,'Selected, slowed replay · consequence banks are read-only in held-out queries; manager still learns · no CPU energy/speed claim.')]:text(d,(115,yy),line,26,MUTED)
        self.features=[];self.units=[]
        # Every displayed small node maps to a real encoded input position.
        for k in range(3):
            count=109 if k==0 else RAW*2 if k==1 else RAW;j=np.arange(count);yy=1-2*(j+.5)/count;rr=np.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5));self.features.append(np.column_stack((-.52+.27*rr*np.cos(theta),.82*yy,.52*rr*np.sin(theta))))
            self.units.append(np.asarray([[.65,.0,0]]) if k==0 else np.asarray([[.44+(j%2)*.37,.82-(j//2)*.52,.16*((j%2)-.5)] for j in range(8)]))
    def choose(self,t):
        if t<5:ids=list(range(0,8))+list(range(24,32));p=t/5;label='OBSERVE → PREDICT → RECEIVE OUTCOME'
        elif t<12:ids=list(range(52,76))+list(range(124,148));p=(t-5)/7;label='DISTRACTORS · MANAGER KEEPS MEMORY CLOSED'
        elif t<22:ids=sum([list(range(i,i+8)) for i in [148,180,212,244]],[]);p=(t-12)/10;label='REPEATING INPUTS · NATIVE PERMISSION OPENS WRITING'
        elif t<26:ids=list(range(276,288));p=(t-22)/4;label='RETURN TO OLD + NEW MEMORIES'
        else:ids=list(range(300,348));p=(t-26)/6;label='NEW ORDER · PREDICT WITHOUT OUTCOME TEACHING'
        return ids[min(len(ids)-1,int(p*len(ids)))],label
    def frame(self,t):
        i,phase=self.choose(t);a=self.a;e=a['events'][i];qi=i-1 if e['teacher'] else i;key=e['key'];y=a['outputs'][qi];event_y=a['outputs'][i];x=np.asarray(a['inputs'][i]);im=self.base.copy();d=ImageDraw.Draw(im)
        text(d,(160,351),phase,32,GREEN if not e['teacher'] else CYAN,True)
        op='Outcome arrives AFTER prediction' if e['teacher'] else 'Prediction returns BEFORE outcome' if e['phase'] not in ['heldout','final_probe','old_probe'] else 'Evaluation query · no outcome sent to consequence banks'
        text(d,(160,410),op,27,MUTED);text(d,(3040,353),f'Event {i+1} / 396',29,INK,True);text(d,(3040,407),'Recorded seed114101',26,MUTED)
        permit=bool(a['inputs'][i][80]) if e['teacher'] else bool(y[80]);text(d,(160,874),'RAW OBSERVATION',25,MUTED,True);text(d,(160,918),f'{key:04b}',51,INK,True)
        text(d,(500,874),'RECENT-KEY MATCH',25,MUTED,True);text(d,(500,918),'YES' if permit else 'NO',45,GREEN if permit else AMBER,True)
        for k,center in enumerate([(500,1205),(1280,1205),(2070,1205)]):
            b=a['banks'][k];after=np.asarray(b['connections'][i]);before=np.asarray(b['connections'][i-1] if i else b['initial']);f=np.asarray(a['features'][k][i]);fp=Scene.project(self,self.features[k],t,center=center,scale=310);up=Scene.project(self,self.units[k],t,center=center,scale=310)
            active=[0] if k==0 and not e['teacher'] and event_y[80] else np.flatnonzero(np.asarray(event_y[72:80] if k==1 else event_y[64:72])).tolist() if k else []
            # Pre-update responses and post-update connections are labelled separately.
            for j in range(len(after)):
                for bit in np.flatnonzero(before[j]):
                    if bit>=len(fp):continue
                    hot=bool(f[bit]);Scene.edge(d,fp[bit],up[j],CYAN if hot and j in active else blend(BG,CYAN,.22 if hot else .1),3 if hot and j in active else 1)
                for bit in np.flatnonzero((after[j]==1)&(before[j]==0)):
                    if bit<len(fp):Scene.edge(d,fp[bit],up[j],GREEN,5)
                for bit in np.flatnonzero((after[j]==0)&(before[j]==1)):
                    if bit<len(fp):Scene.edge(d,fp[bit],up[j],blend(BG,AMBER,.65),3)
            for bit in np.argsort(fp[:,2]):
                xx,yy,fac=fp[bit];r=(4.5 if f[bit] else 2)*fac;d.ellipse((xx-r,yy-r,xx+r,yy+r),fill=CYAN if f[bit] else DIM)
            for j in range(len(up)):
                xx,yy,fac=up[j];changed=bool((after[j]!=before[j]).any());Scene.orb(d,xx,yy,(29 if k==0 else 17)*fac,GREEN if changed else CYAN if j in active else DIM,changed or j in active)
                text(d,(xx+22,yy-15),'M' if k==0 else f'R{([*range(4),*range(24,28)] if k==1 else [*range(4),*range(16,20)])[j]:02}',23,INK if changed or j in active else MUTED,True)
            xx=180+k*790;text(d,(xx,1555),f'+{b["adds"][i]} / −{b["removed"][i]} connections now',28,GREEN if b['adds'][i] else MUTED,True)
            text(d,(xx,1605),'25 shared + 7 learned key connections' if k==0 else f'{b["touched"][i]} / 8 records touched · 32 connections each',24,MUTED)
            if k:text(d,(xx,1650),'LEARN' if e['labels'][k]>=0 else 'READ ONLY',27,CYAN if e['labels'][k]>=0 else GREEN,True)
            else:text(d,(xx,1650),'LEARN RECENCY' if e['manager_learn'] else 'HOLD STATE',27,CYAN if e['manager_learn'] else MUTED,True)
        # This is an explicitly schematic routed bus, not a direct neuron edge.
        d.line([(720,1430),(810,1475),(1655,1475),(1740,1430)],fill=GREEN if permit else DIM,width=4)
        q=720+(t*130)%1020;d.ellipse((q-4,1471,q+4,1479),fill=GREEN if permit else DIM)
        text(d,(845,1493),'Teacher-write permission · fixed forwarding path',25,GREEN if permit else MUTED)
        text(d,(180,1720),'Green: stored connection added   Amber: removed   Cyan: returned response · geometry is schematic',25,MUTED)
        text(d,(2600,550),'PREDICTION → ACTUAL CONSEQUENCE',29,CYAN,True)
        for k in range(4):
            xx=2600+k*100;v=(key>>(3-k))&1;d.rounded_rectangle((xx,615,xx+77,684),radius=12,fill=(23,40,53),outline=CYAN,width=2);text(d,(xx+25,620),v,44,CYAN,True)
        text(d,(3040,625),f'raw key {key:04b}',31,INK,True)
        actual=str(OUTCOME[key]);shown=e['teacher'] or e['phase'] in ['heldout','final_probe','old_probe']
        text(d,(2600,735),'CONTROLLED NATIVE',25,CYAN,True);text(d,(3040,735),'ACTUAL',25,MUTED,True)
        text(d,(2600,778),prediction(y),54,GREEN if shown and prediction(y)==actual else INK,True);text(d,(3040,778),actual if shown else 'not delivered',40,GREEN if shown else MUTED,True)
        for yy,arm,label in [(875,'open','Always-open control'),(930,'frozen','Trained then frozen'),(985,'manager_frozen','Manager frozen')]:
            text(d,(2600,yy),label,27,MUTED);text(d,(3370,yy),prediction(self.data['arms'][arm]['outputs'][qi]),32,AMBER,True)
        text(d,(2600,1050),'Permission for a following teacher, if one arrives:',27,MUTED);text(d,(2600,1095),'WRITE PERMITTED · READING REMAINS AVAILABLE' if permit else 'WRITE BLOCKED · READING REMAINS AVAILABLE',28,GREEN if permit else AMBER,True)
        conn=np.asarray(a['banks'][2]['connections'][i]);prior=np.asarray(a['banks'][2]['connections'][i-1] if i else a['banks'][2]['initial']);changed=np.flatnonzero((conn!=prior).any(1));active=np.flatnonzero(np.asarray(y[64:72]));focus=int(changed[0]) if len(changed) else int(active[0]) if len(active) else 0
        text(d,(2600,1185),f'PROTECTED R{P1[focus]:02} · MEASURED CONNECTION RAM',26,INK,True)
        for bit in range(RAW):
            xx=2600+(bit%32)*28;yy=1240+(bit//32)*27;old,new=prior[focus,bit],conn[focus,bit];col=GREEN if new and not old else AMBER if old and not new else CYAN if new else (24,37,50)
            d.rounded_rectangle((xx,yy,xx+20,yy+17),radius=3,fill=col)
        text(d,(2600,1595),'384 key-input positions · 32 stored connections',25,MUTED)
        text(d,(2600,1640),'— means no returned outcome response.',25,MUTED)
        text(d,(2600,1685),'The admission rule is designed, not self-invented.',25,MUTED)
        text(d,(2600,1730),'CPU can reproduce the behavior; efficiency unmeasured.',24,MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('bootstrap',2.2),('distractors',8.3),('learning',13.7),('recall',24.3),('heldout',28.2)]:film.frame(t).save(a.out/(name+'.png'))
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','20M','-profile:v','high','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-learning-controls-learning-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(DURATION*FPS):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=DURATION*FPS,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-learning-controls-learning-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-learning-controls-learning-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(status='AUDITED_NATIVE_SYNAPTIC_CONTROLLER_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
