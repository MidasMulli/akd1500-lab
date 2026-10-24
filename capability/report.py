"""Standard scientific plots; preserve counter scope and actual CPU comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from capability.design import P
from capability.metrics import derive


def main():
    m=derive();v=P/'visual';(v/'native-primitives-metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.facecolor':'#101e2d','figure.facecolor':'#101e2d','text.color':'#edf3f9','axes.labelcolor':'#a6b9cc','xtick.color':'#a6b9cc','ytick.color':'#a6b9cc','axes.edgecolor':'#385064','legend.facecolor':'#101e2d','legend.edgecolor':'#385064'})
    fig,ax=plt.subplots(figsize=(10,4),dpi=200)
    for kind,density,color,style,label in [('cnp',256,'#73c9ee','-','CNP / 256 active bits'),('fnp',256,'#efbd72','-','FNP / 256 active bits'),('cnp',512,'#a2afbd','--','CNP / 512 active bits')]:
        rows=sorted([r for r in m['fixed'] if r['kind']==kind],key=lambda r:r['units']);x=[r['units'] for r in rows];y=[next(g['cycles'] for g in r['groups'] if g['density']==density) for r in rows]
        ax.plot(x,y,style,color=color,marker='o',lw=2,label=label)
    ax.set_xscale('log',base=2);ax.set_xticks([128,256,512,1024,2048],['128','256','512','1,024','2,048']);ax.set_ylim(0,46000);ax.set_yticks([0,10000,20000,30000,40000],['0','10k','20k','30k','40k']);ax.set_xlabel('Detector outputs / same 512-bit input');ax.set_ylabel('DMA buffer-service cycles');ax.legend(fontsize=11,loc='upper left');ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(v/'native-primitives-scaling.png',facecolor=fig.get_facecolor());fig.savefig(v/'native-primitives-scaling.svg',facecolor=fig.get_facecolor());plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),dpi=200)
    for mode,color,label in [('cpu','#78d0b6','CPU packed NEON / input packing included'),('cnp','#73c9ee','Native CNP / serial host path'),('fnp','#efbd72','Native FNP / serial host path')]:
        if mode=='cpu':
            rr=sorted(m['cpu'],key=lambda r:r['units']);x=[r['units'] for r in rr];y=[next(g['median_wall_ns']/1000 for g in r['groups'] if g['density']==256 and g['method']=='packed_neon') for r in rr]
        else:
            rr=sorted([r for r in m['fixed'] if r['kind']==mode],key=lambda r:r['units']);x=[r['units'] for r in rr];y=[next(g['wall_us'] for g in r['groups'] if g['density']==256) for r in rr]
        ax.plot(x,y,'-o',color=color,label=label,lw=2)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([128,256,512,1024,2048],['128','256','512','1,024','2,048']);ax.set_yticks([.5,1,2,5,10,20,50,100],['0.5','1','2','5','10','20','50','100']);ax.set_xlabel('Detector outputs / 256 of 512 input bits active');ax.set_ylabel('Microseconds per input / log scale');ax.legend(fontsize=10,loc='upper left');ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(v/'native-primitives-cpu.png',facecolor=fig.get_facecolor());fig.savefig(v/'native-primitives-cpu.svg',facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':main()
