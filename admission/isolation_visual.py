"""Measured qualification and failed capacity gate, with the same zero baseline."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from admission.isolation_archive import BASE,audit
from admission.run import sha
P=Path(__file__).resolve().parents[1]
def main():
    r=audit();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#e7f0f8','axes.labelcolor':'#a2b4c6','xtick.color':'#a2b4c6','ytick.color':'#a2b4c6','axes.edgecolor':'#273a4c','axes.facecolor':'#0b1420','figure.facecolor':'#070c14','savefig.facecolor':'#070c14'})
    fig,axes=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[1,1.15]});fig.subplots_adjust(left=.07,right=.97,bottom=.33,top=.73,wspace=.30)
    fig.text(.065,.93,'Two memories adapt independently. Capacity is still a limit.',fontsize=25,weight='bold')
    fig.text(.065,.875,'AKD1500 · two resident plastic banks · shared teaching clocks · recorded response and RAM readbacks',fontsize=13,color='#a2b4c6')
    labels=['Native\ncorrection','Retirement\nremoved','Trained\nthen frozen','CPU\ndictionary'];colors=['#66cfed','#edb168','#8ba1b5','#70e7b5']
    values=[r['primary'][a]['correct'] for a in ['native','no_retire','frozen']]+[r['cpu']['dictionary']['correct']]
    ax=axes[0];bars=ax.bar(range(4),np.asarray(values)/6912*100,color=colors,width=.62)
    ax.set_title('Alternating rule changes',loc='left',pad=27,weight='bold',fontsize=16)
    ax.set_xticks(range(4),labels);ax.set_ylabel('Correct read-only probes (%)');ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',color='#273a4c',alpha=.5);ax.set_axisbelow(True)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f'{value:,}/6,912',ha='center',fontsize=10)
    ax=axes[1];arms=['native','no_retire','frozen'];x=np.arange(3);aa=[r['pressure'][a]['by_context']['0']['correct'] for a in arms];bb=[r['pressure'][a]['by_context']['1']['correct'] for a in arms]
    ax.bar(x-.18,np.asarray(aa)/96*100,width=.34,color='#edb168',label='A: overloaded bank');ax.bar(x+.18,np.asarray(bb)/96*100,width=.34,color='#66cfed',label='B: untouched neighbour')
    ax.axhline(87/96*100,color='#8ba1b5',linestyle='--',linewidth=1);ax.text(2.47,87/96*100-6,'A gate:\n87/96',ha='right',fontsize=10,color='#e7f0f8',bbox=dict(facecolor='#0b1420',edgecolor='none',pad=3))
    ax.set_title('First-seed pressure confirmation · failed useful gate',loc='left',pad=27,weight='bold',fontsize=15);ax.set_xticks(x,labels[:3]);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',color='#273a4c',alpha=.5);ax.set_axisbelow(True);ax.legend(loc='lower left',frameon=True,facecolor='#0b1420',edgecolor='none',framealpha=1,fontsize=10)
    for i,(a,b) in enumerate(zip(aa,bb)):
        ax.text(i-.18,a/96*100+3,f'{a}/96',ha='center',fontsize=10);ax.text(i+.18,b/96*100+3,f'{b}/96',ha='center',fontsize=10)
    for ax in axes:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    notes=['8,400 foreign-context RAM boundaries unchanged; 24 complete original record payloads retained.',
           'Mixed and solo visible RAM trajectories match at all 700 own-context boundaries per bank (first seed).',
           'Three initializations + three first-seed order replications; not a full factorial. Pressure stops after this pilot.',
           'Host supplies cues, context, external outcomes and fixed roles. Native routing selects the receiving bank.',
           'Seven sequential FNP3s; 16 plastic records. CPU also passes correction and does better under pressure.',
           'No electrical concurrency, general eviction, autonomous scheduling, calibrated speed or energy advantage.']
    for i,note in enumerate(notes):fig.text(.065,.235-i*.031,note,fontsize=11,color='#a2b4c6')
    out=P/'visual/isolated-correction';fig.savefig(out.with_suffix('.png'),dpi=240);fig.savefig(out.with_suffix('.svg'));plt.close(fig)
    (P/'visual/ISOLATION_SOURCE_BINDING.json').write_text(json.dumps(dict(archive_sha256=r['archive_sha256'],renderer_sha256=sha(__file__),outputs={p.name:sha(p) for p in [out.with_suffix('.png'),out.with_suffix('.svg')]}),indent=2)+'\n')
if __name__=='__main__':main()
