"""Source-bound scientific figure, 4K replay still and optional 4K60 MP4."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parents[1]
BG='#0a1019';PANEL='#111c2a';INK='#e3edf7';MUTED='#8fa5b9';BLUE='#5ab8f4';GOLD='#efbc67'
def plot(z,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#34465a','figure.facecolor':BG,'axes.facecolor':BG})
    fig=plt.figure(figsize=(19.2,10.8),dpi=200);a=fig.add_axes([.22,.30,.32,.42]);b=fig.add_axes([.64,.30,.30,.42])
    names=['n128_u8','n512_u64','n512_u512'];labels=['128 inputs / 8 neurons','512 inputs / 64 neurons','512 inputs / 512 neurons'];cpu=[];native=[]
    for cell in names:
        rows=[r for r in z['runs'] if r['run'].startswith('confirm_') and r['fixture']==cell]
        cpu.append(statistics.median(r['cpu']['mean_us_per_sample'] for r in rows));native.append(statistics.median(r['native']['mean_us_per_sample'] for r in rows))
    for i in range(3):
        a.plot([cpu[i],native[i]],[i,i],color='#33465b',lw=3);a.scatter(cpu[i],i,s=130,c=BLUE,label='CPU reference' if i==0 else None);a.scatter(native[i],i,s=130,c=GOLD,marker='s',label='Native + host interface' if i==0 else None)
        a.text(cpu[i],i-.19,f'{cpu[i]:.2f} µs',color=BLUE,ha='center',fontsize=13);a.text(native[i],i+.20,f'{native[i]:.1f} µs',color=GOLD,ha='center',fontsize=13)
    a.set_xscale('log');a.set_xlim(.2,2000);a.set_ylim(-.55,2.55);a.set_yticks(range(3),labels);a.set_xlabel('Mean latency per input + learning update (µs, log scale)',labelpad=20);a.grid(axis='x',color='#243243',alpha=.7);a.legend(loc='upper left',bbox_to_anchor=(0,1.18),frameon=False,fontsize=13)
    r=next(r['native'] for r in z['runs'] if r['run']=='resident_large');vals=[r['setup_ns'],r['release_to_observed_completion_ns'],r['output_drain_ns']];total=sum(vals)
    colors=['#6b87a3',BLUE,GOLD];bottom=0
    for v,c,l in zip(vals,colors,['Input + descriptor setup','Release → observed completion','Output drain']):
        b.barh([0],[v/total*100],left=bottom,color=c,height=.3,label=f'{l}: {v/64/1000:.1f} µs / input');bottom+=v/total*100
    b.set_xlim(0,100);b.set_ylim(-.35,.6);b.set_yticks([]);b.set_title('Resident burst: output drain dominates\nShare of total transaction time (%)',loc='left',pad=25,fontsize=15)
    for i,(v,c,l) in enumerate(zip(vals,colors,['Input + descriptor setup','Release → observed completion','Output drain'])):
        fig.text(.64,.23-i*.035,f'{l}: {v/64/1000:.1f} µs / input',color=c,fontsize=12)
    for ax in [a,b]:ax.spines[['top','right']].set_visible(False)
    fig.text(.055,.93,'MEASURED ON ONE AKD1500',color=MUTED,fontsize=14,weight='bold');fig.text(.055,.865,'CPU reference is faster in these tested round trips',fontsize=27,weight='bold')
    fig.text(.055,.12,'32-NPU mesh integrity passed separately: fixed routing stages + one native learner. All 64 measured updates verified.',fontsize=14)
    fig.text(.055,.075,'Three paired timing repeats per cell. Dynamic clocks; no clock changes. Kernel latency includes PCIe/MMIO and busy host polling.',color=MUTED,fontsize=12)
    fig.text(.055,.04,'No card-energy measurement or peak-utilization claim. Resident completion interval is not core-only silicon latency.',color=MUTED,fontsize=12)
    fig.savefig(out/'measurement-results-4k.png',dpi=200);fig.savefig(out/'measurement-results.svg');plt.close(fig)
    svg=out/'measurement-results.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
def font(size,bold=False):
    for path in ['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)
def replay(data,out,video):
    W,H=3840,2160;base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base);f={n:font(n) for n in [30,34,38,42,48,56,72]};bold=font(80,True)
    d.text((150,110),'AKD1500 / HARDWARE EVIDENCE',font=f[38],fill=MUTED)
    d.text((150,180),'Across all 32 NPUs',font=bold,fill=INK)
    d.text((150,295),'31 fixed routing stages → one native learning layer',font=f[48],fill=MUTED)
    positions={};nodes={}
    # Logical 3x3 coordinate mesh; the upper-left gateway is not a compute node.
    for col in range(1,4):
        for row in range(1,4):
            x=150+(col-1)*690;y=460+(row-1)*450
            d.rounded_rectangle((x,y,x+620,y+390),radius=24,fill=PANEL,outline='#2a3c50',width=3)
            if (col,row)==(1,1):
                d.text((x+35,y+38),'INPUT GATEWAY',font=f[38],fill=MUTED);d.text((x+35,y+110),'128 binary features',font=f[42],fill=INK);d.text((x+35,y+188),'One mapped pass',font=f[38],fill=BLUE);continue
            d.text((x+28,y+22),f'NODE  ({col}, {row})',font=f[34],fill=MUTED)
            for npid in range(4):
                xx=x+25+(npid%2)*295;yy=y+94+(npid//2)*138;positions[(col,row,npid)]=(xx,yy)
                d.rounded_rectangle((xx,yy,xx+275,yy+116),radius=15,fill='#172638',outline='#35506a',width=2)
                d.text((xx+20,yy+14),f'NPU {npid}',font=f[34],fill=INK)
                learner=(col,row,npid)==(3,3,0);d.text((xx+20,yy+61),'LEARNER' if learner else 'fixed route',font=f[30],fill=GOLD if learner else MUTED)
    d.rounded_rectangle((2350,460,3690,1740),radius=25,fill=PANEL,outline='#2a3c50',width=3)
    d.text((2410,500),'MEASURED AT THE LEARNER',font=f[38],fill=MUTED)
    d.text((2410,655),'Match scores / 64 configured neurons',font=f[34],fill=MUTED)
    d.text((2410,1090),'Connection edits from RAM readback',font=f[38],fill=INK)
    changed_total=len({e['neuron'] for s in data['samples'] for e in s['connection_edits']})
    replacement_total=sum(len(e['added']) for s in data['samples'] for e in s['connection_edits'])
    d.text((2410,1415),f'{changed_total} of 64 neuron records changed',font=f[42],fill=INK)
    d.text((2410,1490),f'{replacement_total:,} connection replacements',font=f[42],fill=GOLD)
    d.text((2410,1585),'64 observations · exact independent check',font=f[34],fill=MUTED)
    d.text((150,1875),'RECORDED HARDWARE REPLAY  /  synthetic binary-association stress test',font=f[38],fill=INK)
    d.text((150,1940),'Logical placement and motion are schematic. Internal electrical firing and router timing are not measured.',font=f[34],fill=MUTED)
    d.text((150,2000),'32 mapped NPUs ≠ peak utilization. One plastic layer; power measurement awaits a calibrated sensor.',font=f[34],fill=MUTED)
    layers=data['mapping']['layer_assignments'];stage_nps=[tuple(c['np']) for layer in layers for c in layer['components']]
    def frame(sample,progress):
        im=base.copy();q=ImageDraw.Draw(im);s=data['samples'][sample];changed={x['neuron'] for x in s['connection_edits']}
        npid=stage_nps[min(len(stage_nps)-1,int(progress*len(stage_nps)))];x,y=positions[npid]
        q.rounded_rectangle((x-4,y-4,x+279,y+120),radius=17,outline=BLUE,width=8)
        q.text((150,1750),f'Compiled stage-order replay: {min(32,int(progress*32)+1):02d} / 32',font=f[34],fill=BLUE)
        q.text((2410,565),f'Observation {sample+1:02d} / 64   ·   label {s["label"]}',font=f[56],fill=INK)
        for j,score in enumerate(s['scores']):
            xx=2410+j*18;hh=score*9;q.rectangle((xx,1010-hh,xx+12,1010),fill=GOLD if j in changed else BLUE)
        q.line((2410,1012,3570,1012),fill='#52677d',width=2)
        q.text((3580,1000),'0',font=f[30],fill=MUTED);q.text((3580,715),'32',font=f[30],fill=MUTED)
        edits=sum(len(x['added']) for x in s['connection_edits'])
        q.text((2410,1165),f'{edits} removed  +  {edits} added',font=f[48],fill=GOLD)
        for j in range(64):
            xx=2435+(j%16)*68;yy=1250+(j//16)*24;q.ellipse((xx,yy,xx+13,yy+13),fill=GOLD if j in changed else '#3b526a')
        q.text((2410,1350),'Amber marks the actual changed record',font=f[30],fill=MUTED)
        return im
    frame(63,.999).save(out/'mesh32-replay-4k.png')
    if video:
        args=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','16M','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'mesh32-replay-4k60.mp4')]
        pr=subprocess.Popen(args,stdin=subprocess.PIPE)
        try:
            for i in range(720):
                progress=i/720;sample=min(63,int(progress*64));pr.stdin.write(frame(sample,progress).tobytes())
            pr.stdin.close();assert pr.wait()==0
        finally:
            if pr.poll() is None:pr.terminate();pr.wait()
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();out=P/'visual';out.mkdir(exist_ok=True)
    z=json.loads((P/'results/2026-10-23.json').read_text());data=json.loads((out/'mesh32-data.json').read_text());plot(z,out);replay(data,out,a.video)
    (out/'SOURCE_BINDING.json').write_text(json.dumps(dict(results_sha256=hashlib.sha256((P/'results/2026-10-23.json').read_bytes()).hexdigest(),replay_sha256=hashlib.sha256((out/'mesh32-data.json').read_bytes()).hexdigest(),geometry='Logical schematic',motion='Illustrative replay, not electrical telemetry',video='3840x2160, 60 fps, 12 seconds if generated'),indent=2)+'\n')
if __name__=='__main__':main()
