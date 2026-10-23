"""Measured output-contract comparison and a source-bound paired native replay."""
from pathlib import Path
import argparse,hashlib,json,statistics,subprocess
import numpy as np
from PIL import Image,ImageDraw
from visualize import font,BG,PANEL,INK,MUTED,BLUE,GOLD
P=Path(__file__).resolve().parents[1]

def rows(z,contract):
    family='20261023-class-reduction' if contract==2 else '20261023-compact-readout'
    return [r for r in z['runs'] if r['study']==family and r['name'].startswith('confirm_') and r['output_contract']==contract and r['results']['native']['mode']=='native']

def figure(z,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#34465a','figure.facecolor':BG,'axes.facecolor':BG})
    fig=plt.figure(figsize=(19.2,10.8),dpi=200);a=fig.add_axes([.22,.28,.30,.47]);b=fig.add_axes([.65,.28,.29,.47])
    labels=['512 integer scores','512 threshold responses','Two group-support counts']
    data=np.load(P/'results/2026-10-23-readout.npz',allow_pickle=False)
    for c in range(3):
        rr=rows(z,c);cpu=[r['results']['cpu']['mean_us_per_input'] for r in rr];native=[r['results']['native']['mean_us_per_input'] for r in rr]
        y=2-c;a.plot([statistics.median(cpu),statistics.median(native)],[y,y],color='#34465a',lw=3)
        for vals,color,marker,dy in [(cpu,BLUE,'o',-.20),(native,GOLD,'s',.20)]:
            a.plot([min(vals),max(vals)],[y,y],color=color,lw=4);a.scatter(statistics.median(vals),y,c=color,marker=marker,s=110)
            a.text(statistics.median(vals),y+dy,f'{statistics.median(vals):.1f} µs',ha='center',color=color,fontsize=13)
        phases=[]
        for phase in ['input','submit','fetch']:
            phases.append(statistics.median(np.mean(data[r['study']+'__'+r['name']+'__'+phase+'_ns'])/1000 for r in rr))
        left=0
        for v,color in zip(phases,['#6b87a3',BLUE,GOLD]):b.barh([y],[v],left=left,height=.32,color=color);left+=v
    a.set_xscale('log');a.set_xlim(10,1600);a.set_ylim(-.55,2.55);a.set_yticks([2,1,0],labels);a.set_xlabel('Complete host transaction per input (µs, log scale)',labelpad=20);a.grid(axis='x',color='#243243')
    a.scatter([],[],c=BLUE,label='CPU / matched contract');a.scatter([],[],c=GOLD,marker='s',label='AKD + host interface');a.legend(loc='upper left',bbox_to_anchor=(-.06,1.21),frameon=False,fontsize=12)
    b.set_xlim(0,1050);b.set_ylim(-.55,2.55);b.set_yticks([2,1,0],['Scores','Mask','Counts']);b.set_xlabel('Host-observed phases (µs, linear scale)',labelpad=20);b.grid(axis='x',color='#243243');b.set_axisbelow(True)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=color,label=label) for label,color in [('Input','#6b87a3'),('Submission',BLUE),('Fetch + completion polling',GOLD)]]
    b.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.21),frameon=False,fontsize=11)
    for ax in [a,b]:ax.spines[['top','right']].set_visible(False)
    fig.text(.055,.93,'AKD1500 / MATCHED LEARNING, DIFFERENT OUTPUT CONTRACTS',color=MUTED,fontsize=13,weight='bold')
    fig.text(.055,.865,'Returning two counts cuts host latency; CPU still leads',fontsize=25,weight='bold')
    fig.text(.055,.17,'Same inputs, labels and native learner. All 10,242 final learner RAM words match with the downstream reducer.',fontsize=14)
    fig.text(.055,.115,'Output payload: 2,048 bytes → 512 bytes → 8 bytes. Thresholds discard score magnitudes; counts also discard neuron identity.',color=MUTED,fontsize=12)
    fig.text(.055,.075,'Three interleaved timing repeats per pair on one chip; markers show medians and ranges of run means. Dynamic clocks; busy host polling.',color=MUTED,fontsize=11)
    fig.text(.055,.04,'Synthetic association kernel, not image-service performance. Direct card energy is unmeasured. No CPU speed or efficiency advantage established.',color=MUTED,fontsize=11)
    fig.savefig(out/'readout-results-4k.png',dpi=200);fig.savefig(out/'readout-results.svg');plt.close(fig)
    f=out/'readout-results.svg';f.write_text('\n'.join(x.rstrip() for x in f.read_text().splitlines())+'\n')

def replay(z,out,video):
    raw=np.load(P/'results/2026-10-23-readout.npz',allow_pickle=False);family='20261023-class-reduction'
    mask=raw[family+'__confirm_0_mask__native_scores'].reshape(256,512)
    counts=raw[family+'__confirm_0_reduced__native_scores'].reshape(256,2)
    assert np.array_equal(mask.reshape(256,2,256).sum(axis=2),counts)
    full=statistics.median(r['results']['native']['mean_us_per_input'] for r in rows(z,0));compact=statistics.median(r['results']['native']['mean_us_per_input'] for r in rows(z,1));reduced=statistics.median(r['results']['native']['mean_us_per_input'] for r in rows(z,2));cpu=statistics.median(r['results']['cpu']['mean_us_per_input'] for r in rows(z,2))
    data=dict(status='PAIRED_NATIVE_REPLAY',sample_indices=list(range(64)),masks=mask[:64].tolist(),counts=counts[:64].tolist(),mapping=dict(learner=[1,3,0],fixed_reducer=[2,1,0]),contract='Sum threshold>7 responses in each 256-record group',timing=dict(full_scores_us=full,mask_us=compact,two_counts_us=reduced,matched_CPU_two_counts_us=cpu),source_runs=['confirm_0_mask','confirm_0_reduced'],geometry='Logical schematic; paired executions, not simultaneous wire telemetry')
    (out/'readout-replay.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    W,H=3840,2160;base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base);fonts={n:font(n) for n in [28,32,36,40,44,48,56,64,72]}
    d.text((150,100),'AKD1500 / PAIRED HARDWARE REPLAY',font=fonts[36],fill=MUTED)
    d.text((150,175),'512 responses. Two counts.',font=font(88,True),fill=INK)
    d.text((150,300),'Native learning bank → fixed downstream sum → compact host output',font=fonts[48],fill=MUTED)
    d.rounded_rectangle((150,455,1790,1495),radius=30,fill=PANEL,outline='#34465a',width=3)
    d.text((210,505),'NPU (1, 3, 0)  /  NATIVE LEARNER',font=fonts[40],fill=INK)
    d.text((210,575),'512 configured neuron records · threshold response > 7',font=fonts[36],fill=MUTED)
    d.text((210,665),'GROUP 0  /  RECORDS 0–255',font=fonts[32],fill=BLUE)
    d.text((210,1030),'GROUP 1  /  RECORDS 256–511',font=fonts[32],fill=GOLD)
    d.text((210,1415),'Bright dot = recorded threshold response; dim dot = zero',font=fonts[32],fill=MUTED)
    d.rounded_rectangle((2235,455,3690,1495),radius=30,fill=PANEL,outline='#34465a',width=3)
    d.text((2300,505),'NPU (2, 1, 0)  /  FIXED REDUCER',font=fonts[40],fill=INK)
    d.text((2300,575),'Two sums computed downstream on the chip',font=fonts[36],fill=MUTED)
    for g,color,yy in [(0,BLUE,745),(1,GOLD,1105)]:
        d.line([(1790,yy+65),(1995,yy+65),(2235,yy+65)],fill='#34465a',width=5)
        d.polygon([(2210,yy+50),(2235,yy+65),(2210,yy+80)],fill=color)
        d.text((2300,yy),f'Group {g} support / 256',font=fonts[40],fill=color)
        d.rounded_rectangle((2300,yy+105,3590,yy+165),radius=12,fill='#243547')
    d.text((2300,1395),'Host receives 2 × int32 = 8 payload bytes',font=fonts[36],fill=INK)
    d.text((150,1570),'MEASURED HOST TRANSACTION PER INPUT + LEARNING UPDATE',font=fonts[36],fill=MUTED)
    for i,(label,value,color) in enumerate([('512 integer scores',full,MUTED),('512 threshold bytes',compact,BLUE),('Two group counts',reduced,GOLD),('CPU / two counts',cpu,INK)]):
        xx=150+i*900;d.rounded_rectangle((xx,1650,xx+840,1855),radius=18,fill=PANEL)
        d.text((xx+30,1680),label,font=fonts[36],fill=color);d.text((xx+30,1745),f'{value:.1f} µs',font=fonts[64],fill=INK)
    d.text((150,1920),'Identical input/label streams and final learner RAM. Thresholds discard magnitudes; counts discard neuron identity.',font=fonts[32],fill=INK)
    d.text((150,1980),'64 of 256 verified observations replayed from paired native runs. Layout/routes are schematic; electrical timing is unmeasured.',font=fonts[32],fill=MUTED)
    d.text((150,2040),'Synthetic association workload. CPU remains faster. Card energy still requires a connected, calibrated sensor.',font=fonts[32],fill=MUTED)
    def frame(i):
        im=base.copy();q=ImageDraw.Draw(im);sample=min(63,int(i/720*64))
        for j,on in enumerate(mask[sample]):
            group=j//256;k=j%256;xx=228+(k%32)*47;yy=745+group*365+(k//32)*35;color=(BLUE if group==0 else GOLD) if on else '#2b4055'
            q.ellipse((xx,yy,xx+16,yy+16),fill=color)
        for group,color,yy in [(0,BLUE,745),(1,GOLD,1105)]:
            count=int(counts[sample,group]);q.text((3290,yy-8),str(count),font=fonts[56],fill=INK)
            q.rounded_rectangle((2300,yy+105,2300+1290*count/256,yy+165),radius=12,fill=color)
        q.text((2300,1320),f'Observation {sample+1:02d} / 64  ·  both outputs verified',font=fonts[36],fill=MUTED)
        return im
    frame(360).save(out/'readout-replay-4k.png')
    if video:
        pr=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','16M','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'readout-replay-4k60.mp4')],stdin=subprocess.PIPE)
        try:
            for i in range(720):pr.stdin.write(frame(i).tobytes())
            pr.stdin.close();assert pr.wait()==0
        finally:
            if pr.poll() is None:pr.terminate();pr.wait()
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();out=P/'visual';z=json.loads((P/'results/2026-10-23-readout.json').read_text());figure(z,out);replay(z,out,a.video)
    sources=[P/'results/2026-10-23-readout.json',P/'results/2026-10-23-readout.npz',out/'readout-replay.json']
    (out/'READOUT_SOURCE_BINDING.json').write_text(json.dumps(dict(status='BOUND_TO_VERIFIED_NATIVE_RECORDS',sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},geometry='Logical two-NPU schematic',activity='Recorded threshold responses and downstream counts from matched separate runs',timing='Temporal replicate medians, host transactions; not electrical wire timing'),indent=2)+'\n')
