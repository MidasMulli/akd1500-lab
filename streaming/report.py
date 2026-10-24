"""Population-level result plots; no selection by visual success."""
import json
import numpy as np
from streaming.archive import BASE
from streaming.design import ROOT,save

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=json.loads(BASE.with_suffix('.json').read_text());pair=meta['pair'];a,b=pair['learn'],pair['frozen']
    phases=['baseline','changed_before','changed_after','retention'];labels=['Original stream','Changed / before feedback','Changed / after feedback','Unchanged context']
    series=[('Native learning','#73c9ee',[a['metrics'][p]['prediction'] for p in phases]),
            ('Trained, then frozen','#efbd72',[b['metrics'][p]['prediction'] for p in phases]),
            ('CPU nearest neighbor','#78d0b6',[a['metrics'][p]['cpu_prediction'] for p in phases]),
            ('Frozen Nano T202','#b69bea',[a['metrics'][p]['nano_prediction'] for p in phases])]
    plt.rcParams.update({'font.size':12,'figure.facecolor':'#101e2d','axes.facecolor':'#101e2d','text.color':'#edf3f9','axes.labelcolor':'#edf3f9','xtick.color':'#a1b5c8','ytick.color':'#a1b5c8','axes.edgecolor':'#30495f','savefig.facecolor':'#101e2d','svg.hashsalt':'akd-streaming-20261024'})
    fig,ax=plt.subplots(figsize=(13,4.3));x=np.arange(4);w=.19
    for j,(name,color,values) in enumerate(series):
        bars=ax.bar(x+(j-1.5)*w,100*np.asarray(values),w,color=color,label=name)
        ax.bar_label(bars,fmt='%.0f%%',fontsize=10,padding=3,color='#edf3f9')
    ax.set_ylim(0,108);ax.set_ylabel('Correct next-band predictions (%)');ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,fontsize=11)
    for i,p in enumerate(phases):ax.text(x[i],-17,f'n={a["metrics"][p]["n"]}',ha='center',color='#a1b5c8',fontsize=10)
    fig.subplots_adjust(left=.07,right=.99,bottom=.18,top=.85)
    out=ROOT/'visual';fig.savefig(out/'streaming-prediction-results.png',dpi=140);fig.savefig(out/'streaming-prediction-results.svg',metadata={'Date':None});plt.close(fig)
    learning=next(r for r in meta['runs'] if r['name']=='s105101_learn'); frozen=next(r for r in meta['runs'] if r['name']=='s105101_frozen')
    cr=[r for r in learning['rows'] if r['phase']=='changed_after'];fr=[r for r in frozen['rows'] if r['phase']=='changed_after'];rt=[r for r in learning['rows'] if r['phase']=='retention'];ft=[r for r in frozen['rows'] if r['phase']=='retention']
    m=dict(status=pair['status'],phases=phases,series=[dict(name=n,color=c,accuracy=v) for n,c,v in series],
           changed=dict(native_correct=sum(r['prediction']==r['truth'] for r in cr),frozen_correct=sum(r['prediction']==r['truth'] for r in fr),cpu_correct=sum(r['cpu_prediction']==r['truth'] for r in cr),nano_correct=sum(r['nano_prediction']==r['truth'] for r in cr),n=len(cr)),
           retention=dict(native_correct=sum(r['prediction']==r['truth'] for r in rt),frozen_correct=sum(r['prediction']==r['truth'] for r in ft),n=len(rt)),
           skipped_inferences=0,skip_shadow=a['skip_shadow'],physical_sessions=len(meta['runs']),
           native_forwards=sum(r['native_forwards'] for r in meta['runs']),samples=sum(r['samples'] for r in meta['runs']))
    save(out/'streaming-prediction-metrics.json',m);return m

if __name__=='__main__':print(json.dumps(main()))
