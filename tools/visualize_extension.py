"""A measured-state network replay and matched host-offload figures."""
from pathlib import Path
import argparse,hashlib,json,statistics,subprocess
import numpy as np
from PIL import Image,ImageDraw
from visualize import font,BG,PANEL,INK,MUTED,BLUE,GOLD
from verify_network import Learner
P=Path(__file__).resolve().parents[1];COLORS=[BLUE,GOLD,'#a29bfe']

def replay(out,video):
    z=json.loads((P/'results/2026-10-23-balanced-network.json').read_text());raw=np.load(P/'results/2026-10-23-balanced-network.npz',allow_pickle=False)
    run=next(r for r in z['runs'] if r['name']=='confirm_96101_all_learning');key=run['fixture_key'];plan=z['fixtures'][key]
    inputs=raw['fixture__'+key+'__inputs'].reshape(-1,512);labels=raw['fixture__'+key+'__labels'];banks=[Learner(raw['fixture__'+key+f'__bank{k}_ram'].tolist()) for k in range(3)]
    frames=[];cumulative=[0,0,0]
    for i in range(64):
        x=sum(int(v)<<k for k,v in enumerate(inputs[i]));records=[]
        for k,b in enumerate(banks):
            before=b.state[:];p=b.forward(x,int(labels[i]),True);r=[int(v>15) for v in p];after=raw[f'{run["name"]}__native_snapshot_{i}_bank{k}_ram'].tolist();assert after==b.state
            changed=[];edits=0
            for j in range(64):
                a=2+j*20+4;old=before[a:a+16];new=after[a:a+16]
                if old!=new:changed.append(j);edits+=sum((int(v)^int(w)).bit_count() for v,w in zip(old,new))
            cumulative[k]+=edits//2;records.append(dict(reconstructed_responses=r,measured_changed_records=changed,measured_connection_replacements=edits//2,cumulative_replacements=cumulative[k],RAM_sha256=hashlib.sha256(np.array(after,dtype='<u4').tobytes()).hexdigest()))
            x=sum(int(v if bit<4 else not v)<<(j*8+bit) for j,v in enumerate(r) for bit in range(8))
        counts=raw[run['name']+'__native_scores'].reshape(64,2)[i].tolist();assert counts==[sum(r[:32]),sum(r[32:])]
        frames.append(dict(sample=i,label=int(labels[i]),banks=records,measured_final_counts=counts))
    data=dict(status='RECORDED_RAM_AND_OUTPUT_REPLAY',run=run['name'],logical_placements=plan['banks'],fixed_placements=plan['fixed_placements'],samples=frames,geometry='Schematic logical layer placement, not physical floorplan',responses='Reconstructed from exact integer reference; not electrical spike telemetry',edges='Compiler-routed fixed balanced 4+4 encoders and final two-count readout; motion is illustrative')
    (out/'network-replay.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    W,H=3840,2160;base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
    d.text((145,90),'AKD1500 / THREE CONNECTED NATIVE LEARNING BANKS',font=font(36),fill=MUTED)
    d.text((145,165),'One stream. Three plastic stages.',font=font(88,True),fill=INK)
    d.text((145,290),'192 configured neurons · 6 mapped NPUs · no host connection writes after programming',font=font(44),fill=MUTED)
    positions=[150,1430,2710]
    for k,xx in enumerate(positions):
        color=COLORS[k];npid=tuple(plan['banks'][k]['np']);d.rounded_rectangle((xx,450,xx+980,1480),radius=34,fill=PANEL,outline='#34465a',width=3)
        d.text((xx+55,500),f'LEARNING BANK {k+1}',font=font(42,True),fill=color)
        d.text((xx+55,565),f'NPU {npid} / 64 neuron records',font=font(34),fill=MUTED)
        d.text((xx+55,1195),'Connection replacements this input',font=font(32),fill=MUTED)
        d.text((xx+55,1340),'Native replacements / running total',font=font(30),fill=MUTED)
        if k<2:
            mid=xx+1130;d.line((xx+990,880,xx+1260,880),fill=color,width=5);d.polygon([(xx+1240,865),(xx+1260,880),(xx+1240,895)],fill=color)
            d.text((mid,750),'FIXED',font=font(28),fill=MUTED,anchor='mm');d.text((mid,795),'4+4 code',font=font(28),fill=MUTED,anchor='mm');d.text((mid,835),str(tuple(plan['fixed_placements'][k])),font=font(26),fill=MUTED,anchor='mm')
    d.text((150,1545),'Dot fill: reconstructed threshold response. White ring: measured connection record changed.',font=font(38),fill=INK)
    d.text((150,1610),'Every bank was read back after every input. The host supplies the input and class label; updates happen natively.',font=font(34),fill=MUTED)
    d.rounded_rectangle((150,1715,3690,1910),radius=24,fill=PANEL)
    d.text((200,1750),f'FIXED READOUT NPU {tuple(plan["fixed_placements"][-1])} / TWO MEASURED GROUP-SUPPORT COUNTS',font=font(34),fill=MUTED)
    d.text((150,1970),'Three seeds passed exact output + per-step RAM checks. Freeze controls: all banks fixed, or only the middle bank fixed.',font=font(32),fill=INK)
    d.text((150,2035),'Recorded replay, time expanded. Routes and neuron locations are schematic. No accuracy, core-speed or electrical-efficiency claim.',font=font(30),fill=MUTED)
    def frame(fi):
        # 64 observations over 12 seconds; the motion does not encode electrical timing.
        idx=min(63,int(fi/720*64));record=frames[idx];im=base.copy();q=ImageDraw.Draw(im)
        q.text((3100,100),f'INPUT {idx+1:02} / 64',font=font(34),fill=INK)
        for k,xx in enumerate(positions):
            r=record['banks'][k];color=COLORS[k]
            for j,on in enumerate(r['reconstructed_responses']):
                x=xx+96+(j%8)*108;y=690+(j//8)*56
                q.ellipse((x-14,y-14,x+14,y+14),fill=color if on else '#2b4055')
                if j in r['measured_changed_records']:q.ellipse((x-23,y-23,x+23,y+23),outline=INK,width=4)
            q.text((xx+55,1240),str(r['measured_connection_replacements']),font=font(64,True),fill=INK)
            q.text((xx+55,1390),str(r['cumulative_replacements']),font=font(48),fill=color)
        c=record['measured_final_counts'];q.text((200,1810),f'Group 0: {c[0]:2} / 32       Group 1: {c[1]:2} / 32',font=font(52),fill=INK)
        q.text((2860,1820),f'Class label: {record["label"]}',font=font(40),fill=MUTED)
        return im
    frame(360).save(out/'network-replay-4k.png')
    if video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','16M','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'network-replay-4k60.mp4')]
        pr=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for i in range(720):pr.stdin.write(frame(i).tobytes())
            pr.stdin.close();assert pr.wait()==0
        finally:
            if pr.poll() is None:pr.terminate();pr.wait()
    return data

def figure(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=json.loads((P/'results/2026-10-23-offload.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#34465a','figure.facecolor':BG,'axes.facecolor':BG})
    fig,axes=plt.subplots(1,2,figsize=(19.2,10.8),dpi=200);fig.subplots_adjust(left=.085,right=.96,bottom=.27,top=.73,wspace=.33)
    for ax,arms,title in zip(axes,[['mmio','dma'],['resident_busy','resident_sleep']],['512 full match scores / identical contract','Two counts / identical queued contract']):
        vals=[z['confirmation'][a] for a in arms];xx=np.arange(2);w=.32
        for off,metric,color,label in [(-w/2,'wall_us_per_input',GOLD,'Host transaction'),(w/2,'calling_thread_cpu_us_per_input',BLUE,'Calling-thread CPU')]:
            v=[r[metric] for r in vals];bars=ax.bar(xx+off,v,width=w,color=color,label=label)
            for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(v)*.025,f'{b.get_height():.1f}',ha='center',color=color,fontsize=16)
        ax.set_xticks(xx,['MMIO','Kernel DMA read'] if arms[0]=='mmio' else ['Busy polling','50 µs sleeps']);ax.set_ylabel('µs per input');ax.set_title(title,pad=28);ax.set_ylim(0,max(vv[m] for vv in vals for m in ['wall_us_per_input','calling_thread_cpu_us_per_input'])*1.23);ax.grid(axis='y',color='#243243');ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper right')
    fig.text(.07,.93,'AKD1500 / MATCHED OUTPUTS + COMPLETE LEARNING RAM',fontsize=13,color=MUTED)
    fig.text(.07,.86,'Less host CPU, with the same native learning result',fontsize=27,weight='bold')
    fig.text(.07,.17,'Full-score DMA: 9.2× less calling CPU at similar host latency. Queued sleep: 2.4× less calling CPU, with about 15% more latency.',fontsize=15)
    fig.text(.07,.11,'Medians of three interleaved run means on one chip. Identical input/label streams within each pair; dynamic clocks.',fontsize=12,color=MUTED)
    fig.text(.07,.065,'Caller CPU is not whole-system CPU or energy. Initial small DMA smoke was much slower and is preserved; cause unresolved.',fontsize=12,color=MUTED)
    fig.text(.07,.03,'Synthetic association workload. No direct card-energy measurement or AKD-over-CPU speed/efficiency advantage established.',fontsize=12,color=MUTED)
    fig.savefig(out/'offload-results-4k.png');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();out=P/'visual';figure(out);replay(out,a.video)
    fs=[P/'results/2026-10-23-balanced-network.json',P/'results/2026-10-23-balanced-network.npz',P/'results/2026-10-23-offload.json',out/'network-replay.json']
    (out/'NETWORK_SOURCE_BINDING.json').write_text(json.dumps(dict(status='BOUND_TO_NATIVE_RAM_AND_FINAL_OUTPUTS',sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in fs},boundary='Intermediate responses derived; RAM edits and final counts measured. Schematic geometry and expanded time.'),indent=2)+'\n')
