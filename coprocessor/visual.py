"""Evidence-bound resource map and comparison, not a die photograph or spike trace."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
from coprocessor.archive import BASE,P,audit_public

def main():
    report=audit_public();m=json.loads(BASE.with_suffix('.json').read_text());rows=report['metrics']
    bg='#0a101a';fg='#e5edf6';muted='#a1b2c8';blue='#72baff';amber='#f0b765';line='#2a3b51'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':fg,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':fg,'font.size':12,'savefig.facecolor':bg})
    fig=plt.figure(figsize=(19.2,10.8),dpi=200,facecolor=bg)
    fig.text(.055,.941,'The PCIe neural pipeline, measured',fontsize=31,weight='bold')
    fig.text(.055,.898,'AKD1500 on Jetson Orin Nano  /  raw RGB to native learning  /  24 October 2026',color=muted,fontsize=15)
    labels=[('HOST INPUT','32 × 32 RGB bytes','No host feature extraction'),('IMAGE CONVERTER','HRC  (0,0,0)','8-bit weights → binary events'),('SPATIAL LAYER','CNP1  (1,2,1)','4-bit kernel → 512 binary features'),('LEARNING BANK','FNP3  (3,3,0)','128 records · native connection edits')]
    for i,(tag,title,detail) in enumerate(labels):
        x=.055+i*.235
        fig.add_artist(FancyBboxPatch((x,.735),.20,.112,boxstyle='round,pad=.006,rounding_size=.008',transform=fig.transFigure,facecolor='#111d2c',edgecolor=line,linewidth=1.3))
        fig.text(x+.012,.817,tag,color=amber if i==3 else blue,fontsize=11,weight='bold')
        fig.text(x+.012,.784,title,fontsize=17,weight='bold')
        fig.text(x+.012,.752,detail,color=muted,fontsize=10.4)
        if i<3:fig.text(x+.218,.787,'→',fontsize=21,color=muted,ha='center')
    fig.text(.055,.693,'One resident compiler pass. The host supplies images and labels; all intermediate features stay on the AKD.',fontsize=14)
    ax=fig.add_axes([.055,.199,.36,.414]);ax.set_facecolor(bg);ax.set_xlim(.45,3.55);ax.set_ylim(3.55,.45);ax.set_aspect('equal');ax.axis('off')
    for col in range(1,4):
        for row in range(1,4):
            ax.add_patch(FancyBboxPatch((col-.43,row-.43),.86,.86,boxstyle='round,pad=.01,rounding_size=.05',facecolor='#111d2c',edgecolor=line))
            if (col,row)==(1,1):
                ax.text(col,row-.13,'DMA',ha='center',color=blue,fontsize=13,weight='bold');ax.text(col,row+.14,'data + config',ha='center',fontsize=9,color=muted);continue
            ax.text(col,row-.32,f'Node ({col},{row})',ha='center',fontsize=10,color=muted)
            for slot in range(4):
                xx=col+(-.20 if slot%2==0 else .20);yy=row+(-.07 if slot<2 else .24)
                active=[col,row,slot] in [[1,2,1],[3,3,0]]
                color=amber if [col,row,slot]==[3,3,0] else blue if active else '#3d5672'
                ax.add_patch(Circle((xx,yy),.115,facecolor=color,edgecolor=fg if active else line,linewidth=1.2))
                ax.text(xx,yy,str(slot),ha='center',va='center',fontsize=10,color=bg if active else fg)
    fig.text(.055,.638,'32 NPU positions · supported roles vary',fontsize=18,weight='bold')
    fig.text(.055,.159,'Highlighted: two NPUs exercised here, plus the separate HRC.',fontsize=11,color=muted)
    fig.text(.055,.132,'Logical resource coordinates. Placement is not a die photograph.',fontsize=11,color=muted)
    fig.text(.515,.638,'Full-pipeline elapsed time',fontsize=18,weight='bold')
    labels=['CPU NEON','AKD mapped I/O','AKD DMA read','AKD DMA read + write'];values=[]
    for learn in [False,True]:
        a=[r for r in rows if r['mode']=='cpu-dense' and r['cell']=='rgb32_c128' and r['learn']==learn][0]['median_us']
        v=[a]
        for profile in [0,1,2]:v.append(float(np.median([r['median_us'] for r in rows if r['name'].startswith('transport_') and r['profile']==profile and r['learn']==learn and r['count']==96])))
        values.append(v)
    ax=fig.add_axes([.65,.25,.295,.335]);ax.set_facecolor(bg);y=np.arange(4)
    for k,(color,label) in enumerate([(blue,'Frozen'),(amber,'Learning')]):
        bars=ax.barh(y+(k-.5)*.28,values[k],height=.25,color=color,label=label)
        for bar,value in zip(bars,values[k]):ax.text(value+11,bar.get_y()+bar.get_height()/2,f'{value:.0f}',va='center',fontsize=11)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,850);ax.set_xlabel('µs per image · lower is faster',labelpad=12);ax.grid(axis='x',color=line,alpha=.7);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(line);ax.tick_params(axis='y',length=0,pad=12);ax.legend(frameon=False,loc='lower right',bbox_to_anchor=(1,1.06),ncol=2,fontsize=11)
    fig.text(.515,.161,'DMA both ways: ~99 µs calling CPU, ~707 µs elapsed.',fontsize=12,color=muted)
    fig.text(.515,.132,'Caller time excludes kernel workers; energy remains unmeasured.',fontsize=11,color=muted)
    fig.text(.055,.079,'3,872 native forwards  ·  1,016 full learner-RAM captures  ·  every score and captured state independently replayed',fontsize=14,weight='bold')
    fig.text(.055,.048,'Seeded characterization fixtures, not a trained vision model. One chip / initialization. Setup + RAM diagnostics excluded from timing.',fontsize=11,color=muted)
    out=P/'visual/pcie-fabric-4k';fig.savefig(out.with_suffix('.png'));fig.savefig(out.with_suffix('.svg'));plt.close(fig)
    svg=out.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    binding=dict(status='RECORDED_EVIDENCE_NOT_LIVE',source_archive_sha256=digest(BASE.with_suffix('.npz')),source_metadata_sha256=digest(BASE.with_suffix('.json')),renderer_sha256=digest(Path(__file__)),outputs={p.name:digest(p) for p in [out.with_suffix('.png'),out.with_suffix('.svg')]},coordinates='SDK resource coordinates; no die-layout or router-hop claim')
    (P/'visual/PCIE_FABRIC_SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
if __name__=='__main__':main()
