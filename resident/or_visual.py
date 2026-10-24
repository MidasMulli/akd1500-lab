"""Measured three-bank repair replay. Response reconstruction and schematic flow labeled."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from resident.or_archive import BASE,expand_run,audit_public
from resident.or_design import P,Network
from tools.visualize import font as existing_font

@lru_cache(maxsize=64)
def font(size,bold=False):return existing_font(size,bold)

BG='#080f19';PANEL='#101e2d';INK='#edf3f9';MUTED='#99afc4';LINE='#2a4259'
COLORS=['#73c9ee','#b69bea','#efbd72','#78d0b6'];GOLD='#f4c771'
SEGMENTS=[(384,448,4),(448,512,4),(512,640,6),(640,704,6),(704,768,4)]
DURATION=sum(sec for a,b,sec in SEGMENTS)


def records():
    audit_public();report=json.loads(BASE.with_suffix('.json').read_text())
    rec=next(r for r in report['runs'] if r['name']=='full_u512_s102101_repair');key=rec['name']+'__'
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        data=expand_run(rec,{k.split('__',1)[1]:z[k] for k in z.files if k.startswith(key)})
        controls={arm:z[f'full_u512_s102101_{arm}__native.scores'] for arm in ['repair','frozen','wrong_bank']}
    net=Network([data[f'bank{k}.initial'].tolist() for k in range(3)],512);events=[];selected=0
    for i,row in enumerate(rec['rows']):
        x=data['stream.inputs'][i];before=[b.ram[:] for b in net.banks];responses=[];bits_x=x.copy()
        for b in net.banks:
            bits=sum(1<<int(j) for j in np.flatnonzero(bits_x));p=np.asarray([(w&bits).bit_count() for w in b.weights]);responses.append(p.tolist())
            presence=(p>27).reshape(4,128).any(1);bits_x=np.repeat(presence,128).astype(np.uint8)
        output,detail=net.forward(x,row['label'],row['mask']);assert output==data['native.scores'][i].tolist()
        banks=[]
        for k,b in enumerate(net.banks):
            after=data[f'bank{k}.snapshots'][i];assert np.array_equal(after,b.ram)
            aa=np.asarray(before[k][2:],np.uint32).reshape(512,20);bb=after[2:].reshape(512,20)
            ids=np.flatnonzero((aa[:,4:]!=bb[:,4:]).any(1)).tolist();added=sum(int(v).bit_count() for v in (bb[:,4:]&~aa[:,4:]).ravel() if v)
            banks.append(dict(edited_records=ids,inserted_connection_bits=added,potentials=responses[k],supports=detail[k]['supports'],presence=detail[k]['presence']))
        if banks[0]['edited_records']:selected=banks[0]['edited_records'][0]
        at=2+selected*20+4;weights=data['bank0.snapshots'][i,at:at+16];prior=np.asarray(before[0][at:at+16],np.uint32)
        addresses=[j for j in range(512) if int(weights[j//32])>>(j%32)&1];new=[j for j in addresses if not(int(prior[j//32])>>(j%32)&1)]
        assert len(addresses)==32
        events.append(dict(**row,banks=banks,measured_output=output,selected_record=selected,selected_addresses=addresses,new_addresses=new))
    v=P/'visual';v.mkdir(exist_ok=True)
    (v/'fnp-or-relay-replay.json').write_text(json.dumps(dict(status='MEASURED_OUTPUT_AND_COMPLETE_RAM_BOUND_REPLAY',seed=102101,events=events[384:],
        allocated_records=1536,learning_nps=rec['fixture_info']['learning_nps'],mapping=rec['fixture_info']['mapping'],
        boundary='Output and full RAM measured; potentials and class-presence messages reconstructed. Geometry and motion schematic; time expanded. Host supplies events and teachers.'),separators=(',',':'))+'\n')
    return report,rec,data,controls,events


def main(video=False):
    report,rec,data,controls,events=records();W,H=3840,2160;base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
    def txt(x,y,text,size=32,color=INK,bold=False,anchor=None):d.text((x,y),text,font=font(size,bold),fill=color,anchor=anchor)
    txt(110,65,'AKD1500 / THREE CONNECTED NATIVE LEARNING MEMORIES',32,MUTED)
    txt(110,125,'Change the signal. Repair one memory. Keep the other two.',66,INK,True)
    txt(110,230,'1,536 allocated neuron records · three plastic NPUs + two fixed encoders + one fixed readout',34,MUTED)
    xs=[110,1390,2670];boardwidth=1060
    for k,(x,title) in enumerate(zip(xs,['RECOGNITION MEMORY','MAPPING MEMORY 1','MAPPING MEMORY 2'])):
        d.rounded_rectangle((x,465,x+boardwidth,1280),radius=24,fill=PANEL,outline=LINE,width=3)
        txt(x+40,500,title,34,COLORS[k],True)
        npid=tuple(rec['fixture_info']['learning_nps'][k]);txt(x+40,555,f'NPU {npid}  /  512 learning-neuron records',27,MUTED)
        for c in range(4):txt(x+25,710+c*112,str(c),26,COLORS[c])
        txt(x+40,1188,'Dots: reconstructed responses > threshold 27',25,MUTED)
        txt(x+40,1230,'White ring: measured connection record changed',25,MUTED)
    for k,x in enumerate([1280,2560]):
        txt(x,510,'FIXED',23,MUTED,anchor='ma');txt(x,545,'OR ENCODER',22,MUTED,anchor='ma')
        txt(x,1050,'Class present',23,MUTED,anchor='ma');txt(x,1088,'→ 128 bits',23,MUTED,anchor='ma')
        placement=rec['fixture_info']['mapping'][1+k*2]['nps'][0]['np'];txt(x,1130,str(tuple(placement)),21,MUTED,anchor='ma')
    d.rounded_rectangle((110,1335,1810,1900),radius=24,fill=PANEL,outline=LINE,width=3)
    d.rounded_rectangle((1880,1335,3730,1900),radius=24,fill=PANEL,outline=LINE,width=3)
    txt(155,1370,'MATCHED HARDWARE CONTROLS / SAME INPUT STREAM',29,MUTED)
    for j,label in enumerate(['Repair recognition','Freeze all three','Teach mapping 2 instead']):txt(160,1505+j*112,label,30,INK)
    txt(160,1850,'Counts accumulate only within the phase currently shown.',25,MUTED)
    txt(1925,1370,'INSIDE THE RECOGNITION RECORD / MEASURED CONNECTIONS',29,MUTED)
    txt(110,1915,'Connection records actually edited: recognition 8/512 · mapping 1 4/512 · mapping 2 4/512 (recorded seed).',26,MUTED)
    txt(110,1965,'Three initialization seeds: repaired new signals 64/64 each; frozen controls 16/64; old retention 64/64.',29,INK)
    txt(110,2020,'Recorded replay, expanded time. Responses reconstructed; layout and flow schematic. Not an electrical spike trace.',27,MUTED)
    txt(110,2070,'Host supplies sensory events, teachers and mode changes. Intermediate neural processing stays on the chip. No CPU/energy advantage claimed.',25,MUTED)
    phases={'old':'KNOWN SIGNALS / all three memories queried without teaching','changed_before':'SENSOR CODE CHANGES / mappings remain fixed',
        'repair':'TEACH RECOGNITION ONLY / downstream memories stay fixed','changed_after':'HELD-OUT NEW SIGNALS / no teaching','retention':'OLD SIGNALS RETURN / no reminder training'}
    def position(t):
        elapsed=0
        for lo,hi,sec in SEGMENTS:
            if t<elapsed+sec:return min(hi-1,lo+int((t-elapsed)/sec*(hi-lo)))
            elapsed+=sec
        return 767
    def frame(fi):
        t=fi/60;idx=position(t);e=events[idx];phase=e['phase'];im=base.copy();q=ImageDraw.Draw(im)
        def tt(x,y,text,size=30,color=INK,bold=False,anchor=None):q.text((x,y),text,font=font(size,bold),fill=color,anchor=anchor)
        q.rounded_rectangle((110,330,3730,415),radius=16,fill='#1a2c40')
        tt(145,350,phases[phase],33,INK,True);tt(3690,354,f'EVENT {idx+1} / 768 · SEED 102101',26,MUTED,anchor='ra')
        for k,x in enumerate(xs):
            b=e['banks'][k];active=bool(e['mask']&(1<<k));color=COLORS[k]
            q.rounded_rectangle((x+40,607,x+1020,660),radius=11,fill='#234357' if active else '#192b3c')
            tt(x+60,616,'NATIVE LEARNING' if active else 'QUERY ONLY',27,color if active else MUTED,True)
            tt(x+1000,619,f'{len(b["edited_records"])} edited this input',25,INK,anchor='ra')
            edits=set(b['edited_records'])
            for j,p in enumerate(b['potentials']):
                xx=x+100+(j%32)*28;yy=700+(j//32)*28;c=j//128;on=p>27
                q.ellipse((xx-7,yy-7,xx+7,yy+7),fill=COLORS[c] if on else '#243a4d')
                if on:q.ellipse((xx-15,yy-15,xx+15,yy+15),outline=COLORS[c],width=2)
                if j in edits:
                    radius=13+2*math.sin(t*9);q.ellipse((xx-radius,yy-radius,xx+radius,yy+radius),outline=INK,width=3)
            tt(x+40,1148,'Supports: '+'  /  '.join(map(str,b['supports'])),27,INK)
        for k,x in enumerate([1280,2560]):
            for c,present in enumerate(e['banks'][k]['presence']):
                yy=730+c*72;col=COLORS[c] if present else '#30475a'
                q.line((x-97,yy,x+97,yy),fill=col,width=6 if present else 2)
                if present:
                    xx=x-90+((t*.85)%1)*180;q.ellipse((xx-7,yy-7,xx+7,yy+7),fill=INK)
                tt(x,yy-31,f'CLASS {c}',19,col,anchor='ma')
        ii=[j for j in range(384,idx+1) if events[j]['phase']==phase]
        for j,(arm,color) in enumerate([('repair',COLORS[0]),('frozen','#8aa0b4'),('wrong_bank',COLORS[1])]):
            correct=sum(int(np.argmax(controls[arm][i]))==events[i]['destination'] for i in ii);fraction=correct/len(ii);yy=1515+j*112
            q.rounded_rectangle((720,yy,1575,yy+35),radius=7,fill='#253c50')
            if correct:q.rounded_rectangle((720,yy,720+855*fraction,yy+35),radius=7,fill=color)
            tt(1755,yy-6,f'{correct}/{len(ii)}',30,INK,anchor='ra')
        tt(160,1450,'Pre-update teaching responses' if phase=='repair' else 'Held-out query responses',25,GOLD if phase=='repair' else MUTED)
        tt(1925,1430,f'Neuron record {e["selected_record"]} · 32 selected input addresses',30,INK,True)
        selected=set(e['selected_addresses']);new=set(e['new_addresses']);input_bits=data['stream.inputs'][idx]
        for bit in range(512):
            xx=1930+(bit%32)*47;yy=1515+(bit//32)*18
            q.rectangle((xx,yy,xx+33,yy+12),fill=GOLD if bit in new else INK if bit in selected else '#355d72' if input_bits[bit] else '#1c3042')
        tt(1925,1830,'Gold: inserted now · White: stored connection · Blue: active sensory input',24,MUTED)
        out=int(np.argmax(e['measured_output']));tt(3675,1850,f'OUTPUT {out}  /  EXPECTED {e["destination"]}',25,COLORS[out],True,anchor='ra')
        q.rectangle((110,2125,3730,2132),fill='#243c50');q.rectangle((110,2125,110+3620*min(1,t/DURATION),2132),fill=COLORS[0])
        return im
    v=P/'visual';frame(3*60).save(v/'fnp-or-relay-4k.png');frame(11*60).save(v/'fnp-or-relay-repair-4k.png');frame(18*60).save(v/'fnp-or-relay-result-4k.png')
    if video:
        target=v/'fnp-or-relay-4k60.mp4';cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-',
            '-an','-c:v','h264_videotoolbox','-b:v','14M','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for fi in range(DURATION*60):proc.stdin.write(frame(fi).tobytes())
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate()
    files=[BASE.with_suffix('.json'),BASE.with_suffix('.npz'),Path(__file__),v/'fnp-or-relay-replay.json',v/'fnp-or-relay-4k.png',v/'fnp-or-relay-repair-4k.png',v/'fnp-or-relay-result-4k.png']
    if video:files.append(v/'fnp-or-relay-4k60.mp4')
    (v/'FNP_OR_RELAY_SOURCE_BINDING.json').write_text(json.dumps(dict(status='NATIVE_OUTPUT_AND_COMPLETE_STATE_BOUND',duration_seconds=DURATION,display_fps=60,
        hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')


if __name__=='__main__':
    import sys
    main('--video' in sys.argv)
