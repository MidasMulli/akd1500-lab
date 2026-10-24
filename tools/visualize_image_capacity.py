"""Real-image examples, actual CPU timings and the preserved compiler stop."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recognition.archive import BASE,audit_public,pooled

P=Path(__file__).resolve().parents[1]
BG,PANEL,INK,MUTED='#09131f','#101f30','#edf4fc','#a6b8cc'
BLUE,GOLD,RED,GREEN='#72caff','#ffc17d','#e49b9b','#8edbb9'


def main():
    audit_public();meta=json.loads(BASE.with_suffix('.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
        'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#3b526b'})
    fig=plt.figure(figsize=(19.2,10.8),dpi=200,facecolor=BG)
    fig.text(.06,.946,'Real images / quality passes, native mapping stops',fontsize=28,weight='bold')
    fig.text(.06,.904,'MNIST exemplar recognition  ·  18,000 CPU classifications  ·  both native compiler failures preserved  ·  zero AKD runs',fontsize=12.5,color=MUTED)
    rects=[[.075,.555,.375,.245],[.565,.555,.36,.245],[.075,.18,.375,.235],[.565,.18,.36,.235]]
    a,b,c,d=[fig.add_axes(rect,facecolor=PANEL) for rect in rects]
    for ax in [a,b,d]:
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',color='#30455b',alpha=.6)
    a.set_title('01  Accuracy, with the failed small bank retained',loc='left',pad=22,weight='bold',fontsize=14)
    for offset,label,color,values in [(-.18,'Raw',BLUE,[81.4,91]),(.18,'Spatial OR',GOLD,[81.7,90])]:
        a.bar(np.arange(2)+offset,values,width=.32,color=color,zorder=3,label=label)
        for x,y in zip(np.arange(2)+offset,values):a.text(x,y+1.4,f'{y:g}%',ha='center',fontsize=11)
    a.axhline(85,color=RED,ls='--',lw=1.2);a.text(.5,98,'85% floor',ha='center',color=RED,fontsize=9)
    a.set(ylim=(0,100),xticks=[0,1],xticklabels=['480 exemplars\nfailed quality','3,200 exemplars\nfresh test passes'],ylabel='Correct classifications (%)')
    a.legend(loc='upper left',bbox_to_anchor=(0,1.10),frameon=False,labelcolor=INK,fontsize=10,ncols=2)
    fig.text(.075,.485,'Different sealed test subsets; not a matched estimate of capacity gain.',color=MUTED,fontsize=9)
    b.set_title('02  A concrete CPU target, with input processing',loc='left',pad=22,weight='bold',fontsize=14)
    for offset,label,color,kind in [(-.18,'Raw',BLUE,'raw'),(.18,'Spatial OR',GOLD,'compacted')]:
        values=[r['median_wall_ns']/1000 for r in meta['cpu_timings'] if r['representation']==kind]
        b.bar(np.arange(3)+offset,values,width=.32,color=color,label=label,zorder=3)
        for x,y in zip(np.arange(3)+offset,values):b.text(x,y+2,f'{y:.1f}',ha='center',fontsize=10)
    b.set(ylim=(0,125),xticks=range(3),xticklabels=['Overlap vector','Scalar packed','NEON packed'],ylabel='Median wall time / image (µs)')
    b.legend(loc='upper left',bbox_to_anchor=(0,1.10),frameon=False,labelcolor=INK,fontsize=10,ncols=2)
    b.text(.03,.96,'Nano/core2 · 3 × 1,000 images per method\nPooling, packing and class decode included',transform=b.transAxes,color=MUTED,fontsize=9,va='top')
    c.set_title('03  Actual inputs and their spatial compaction',loc='left',pad=22,weight='bold',fontsize=14)
    c.set(xticks=[],yticks=[],xlim=(0,1),ylim=(0,1))
    for spine in c.spines.values():spine.set_visible(False)
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        x=z['capacity__test_inputs'];y=z['capacity__test_labels'];raw=np.asarray(meta['capacity']['sealed']['raw']['predictions']);compact=np.asarray(meta['capacity']['sealed']['compacted']['predictions'])
        groups=[('Both correct',(raw==y)&(compact==y)),('Compaction loses',(raw==y)&(compact!=y)),('Compaction helps',(raw!=y)&(compact==y)),('Both wrong',(raw!=y)&(compact!=y))]
        for j,(label,mask) in enumerate(groups):
            idx=int(np.flatnonzero(mask)[0]);left=.075+j*.091
            for row,im in enumerate([x[idx].reshape(32,32),pooled(x[idx:idx+1]).reshape(16,16)]):
                ax=fig.add_axes([left,.25+(1-row)*.072,.075,.067],facecolor=PANEL);ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_axis_off()
            c.text(.095+j*.243,.96,label,ha='center',va='top',fontsize=9,color=MUTED)
            c.text(.095+j*.243,.25,f'true {y[idx]}\nraw {raw[idx]} · OR {compact[idx]}',ha='center',va='top',fontsize=9)
    c.text(.02,.02,'First test example in each stated outcome category · images, not neuron activity',fontsize=8,color=MUTED)
    d.set_title('04  A layer ceiling; not measured chip capacity',loc='left',pad=22,weight='bold',fontsize=14)
    d.barh([0,1],[3200,3200],height=.38,color=[BLUE,GOLD],alpha=.8)
    d.axvline(2048,color=RED,ls='--',lw=1.5)
    d.text(2110,1.53,'Compiler ceiling: 2,048 filters / layer',color=RED,fontsize=10)
    for yy in [0,1]:d.text(3180,yy,'3,200 → rejected',ha='right',va='center',color=BG,fontsize=11,weight='bold')
    d.set(xlim=(0,3500),ylim=(-.65,1.95),yticks=[0,1],yticklabels=['Raw attempt','Compacted attempt'],xlabel='Filters requested in one convolution layer')
    d.grid(False);d.text(.035,.035,'Virtual counts match; mapping fails. No physical session admitted.',transform=d.transAxes,color=MUTED,fontsize=9)
    fig.text(.06,.085,'Useful accuracy and an exact CPU baseline are ready. A partitioned native bank still needs a verified routing/output design.',fontsize=12)
    fig.text(.06,.045,'CPU measurements + compiler rejection only. No new AKD timing, learning, energy or whole-chip capacity result. Original failures remain public.',fontsize=10,color=MUTED)
    outputs=['visual/image-capacity-4k.png','visual/image-capacity.svg','visual/image-capacity-data.csv']
    fig.savefig(P/outputs[0],dpi=200,facecolor=BG);fig.savefig(P/outputs[1],facecolor=BG);plt.close(fig)
    svg=P/outputs[1];svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with (P/outputs[2]).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['representation','method','median_wall_ns','p95_wall_ns','median_thread_cpu_ns'],lineterminator='\n');writer.writeheader()
        for row in meta['cpu_timings']:writer.writerow({k:row[k] for k in writer.fieldnames})
    paths=['results/2026-10-23-image-capacity.json','results/2026-10-23-image-capacity.npz','recognition/data.py','recognition/capacity.py','recognition/prepare.py','recognition/cpu_bench.cpp','recognition/archive.py','tools/visualize_image_capacity.py']+outputs
    (P/'visual/IMAGE_CAPACITY_SOURCE_BINDING.json').write_text(json.dumps(dict(status='BOUND_TO_INDEPENDENT_CPU_IMAGE_RECOUNT_AND_COMPILER_FAILURES',hashes={p:hashlib.sha256((P/p).read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    print(P/outputs[0])


if __name__=='__main__':main()
