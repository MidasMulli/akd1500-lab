"""Receipt-backed 3840x2160 scientific figures; no illustrative hardware signals."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pong.relearn.audit_public import audit

BG='#0b111b';PANEL='#111e2d';INK='#edf2f9';MUTED='#a2b3c7';BLUE='#71b9e8';GOLD='#edbd6b';PURPLE='#b69ae9';GREY='#8494a8'


def setup():
    plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':PANEL,'savefig.facecolor':BG,
        'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
        'axes.edgecolor':'#334456','grid.color':'#334456','font.family':'DejaVu Sans','font.size':15,
        'axes.titlesize':19,'axes.titleweight':'bold','axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14})


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.4);ax.set_axisbelow(True)


def learning():
    audit();report=json.loads((ROOT/'results/2026-10-23-pong-relearning.json').read_text());arms=report['triple']['arms']
    fig=plt.figure(figsize=(20,11.25),dpi=192)
    fig.text(.055,.935,'AKD1500 / RELEARNING DIAGNOSIS / 23 OCT 2026',color=MUTED,size=16)
    fig.text(.055,.875,'Learning history changes the outcome.',size=32,weight='bold')
    fig.text(.055,.827,'Same new-rule teaching. Separate development trajectories. One physical seed and one chip.',color=MUTED,size=17)
    ax=fig.add_axes([.075,.285,.51,.46]);style(ax)
    colors={'fresh':BLUE,'relearn':GOLD,'frozen':GREY};labels={'fresh':'New-rule-only A','relearn':'Old A → new A','frozen':'Old A → frozen'}
    marks={'fresh':'o','relearn':'s','frozen':'^'}
    for arm in ['fresh','relearn','frozen']:
        ys=[arms[arm]['curves'][str(k)]['new_A']['accuracy']*100 for k in range(7)]
        ax.plot(range(7),ys,color=colors[arm],marker=marks[arm],lw=3,ms=7,label=labels[arm])
    cpu=[arms['relearn']['cpu_baseline'][str(k)]['new_A']['accuracy']*100 for k in range(7)]
    ax.plot(range(7),cpu,color=PURPLE,marker='D',ls='--',lw=2.5,ms=6,label='CPU recent-neighbour learner')
    ax.axhline(65,color=MUTED,ls=':',lw=1.6);ax.text(.1,66.5,'65% gate',color=MUTED,size=12)
    ax.set(xlim=(-.1,6.15),ylim=(0,102),xticks=range(7),yticks=range(0,101,20),xlabel='Passes of 64 new-rule teaching examples',ylabel='Correct landing band on 64 separate trajectories (%)')
    ax.set_title('Prediction after the physics changes',loc='left',pad=20)
    ax.legend(loc='upper left',bbox_to_anchor=(.01,.95),frameon=False,fontsize=12,labelcolor=INK)
    b=fig.add_axes([.68,.285,.255,.46]);style(b)
    entries=[('New-rule-only A','fresh',46),('Old A → new A','relearn',28),('Old A → frozen','frozen',12),('CPU learner','cpu',44)]
    for i,(name,arm,count) in enumerate(entries):
        y=3-i;b.barh(y,count/64*100,color=colors.get(arm,PURPLE),height=.54)
        b.text(count/64*100+2,y,f'{count}/64',va='center',size=17,weight='bold')
    b.set(yticks=range(4),yticklabels=[e[0] for e in entries][::-1],xlim=(0,100),xlabel='Correct (%)')
    b.set_title('Final prediction score',loc='left',pad=20);b.grid(axis='x',alpha=.3);b.grid(axis='y',visible=False)
    fig.text(.075,.192,'Untouched court B: 47/64 before and after adaptation, with zero further B teaching.',size=18,weight='bold')
    fig.text(.075,.143,'Fit on repeated new-rule teaching cases: new-rule-only 63/64 · relearning 54/64.',color=MUTED,size=16)
    fig.text(.075,.097,'Native learning history impairs this case; the gap does not identify one unique stale-weight or hidden-state cause.',color=MUTED,size=14)
    fig.text(.075,.06,'B uses a separate input-code region. No multi-seed native qualification, learned paddle control or substrate advantage is established.',color=MUTED,size=13)
    fig.savefig(ROOT/'visual/pong-relearning-4k.png');plt.close(fig)


def costs():
    report=json.loads((ROOT/'results/2026-10-23-pong-substrate.json').read_text())
    follow=json.loads((ROOT/'results/2026-10-23-pong-read-dma.json').read_text());s={**report['summary'],**follow['summary']}
    kinds=['mmio','dma','read_dma','queued_dma','queued_dma_sleep','queued_read_dma_sleep']
    names=['Serial MMIO','Serial DMA read + write','Serial read-only DMA','Queued DMA read + write','Queued DMA + wait','Queued read-only + wait']
    all_rows=report['rows']+follow['rows']
    cpu_wall=np.mean([r['exact_CPU_elapsed_us_per_input'] for r in all_rows]);cpu_thread=np.mean([r['exact_CPU_thread_us_per_input'] for r in all_rows])
    fig=plt.figure(figsize=(20,11.25),dpi=192)
    fig.text(.055,.935,'AKD1500 / MATCHED RECORDED-PONG COST / 23 OCT 2026',color=MUTED,size=16)
    fig.text(.055,.875,'The host path improves. CPU replay still wins.',size=31,weight='bold')
    fig.text(.055,.827,'Identical neural rule, scores and final RAM. 2,240 inputs/run · three repeats/mode · recorded workload.',color=MUTED,size=17)
    for panel,(key,cpu,title) in enumerate([
        ('native_elapsed_us_per_input',cpu_wall,'Elapsed cost per input'),
        ('native_thread_cpu_us_per_input',cpu_thread,'Calling-thread CPU cost per input')]):
        ax=fig.add_axes([.14+.47*panel,.29,.335,.455]);style(ax)
        values=[s[k][key]['mean'] for k in kinds]+[cpu];count=len(values)
        errors=np.array([[s[k][key]['mean']-s[k][key]['min'] for k in kinds]+[0],
            [s[k][key]['max']-s[k][key]['mean'] for k in kinds]+[0]])
        ax.barh(range(count),values,color=[GREY,BLUE,BLUE,BLUE,BLUE,GOLD,PURPLE],height=.56,xerr=errors,
            error_kw={'ecolor':INK,'capsize':4,'lw':1.5})
        for y,value in enumerate(values):ax.text(value+errors[1,y]+9,y,f'{value:.1f} µs',va='center',weight='bold',size=16)
        ax.set(yticks=range(count),yticklabels=names+['CPU exact-rule replay'],xlim=(0,535),xlabel='Microseconds / input · lower is better')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.3);ax.grid(axis='y',visible=False);ax.set_title(title,loc='left',pad=20)
        if panel:ax.set_yticklabels([])
    reduced=1-s['queued_read_dma_sleep']['native_thread_cpu_us_per_input']['mean']/s['mmio']['native_thread_cpu_us_per_input']['mean']
    fig.text(.075,.198,f'Queued waiting uses {reduced*100:.0f}% less caller CPU than MMIO; exact CPU replay still uses less.',size=18,weight='bold')
    fig.text(.075,.15,'Queued elapsed/input is batch-amortized cost, not individual response latency. Whiskers show the range of three runs.',color=MUTED,size=15)
    fig.text(.075,.107,'Excludes programming, feature generation and verification. CPU replay includes input packing, checksums and repeated state initialization.',color=MUTED,size=14)
    fig.text(.075,.065,'Calling-thread CPU excludes system/IRQ work. No energy measurement, fresh-game batching or silicon-peak claim.',color=MUTED,size=14)
    fig.text(.075,.032,'Read-only DMA is a separately frozen follow-up chosen after the four-mode primary test; all raw runs are preserved.',color=MUTED,size=13)
    fig.savefig(ROOT/'visual/pong-substrate-4k.png');plt.close(fig)


if __name__=='__main__':setup();learning();costs();print('PLOT_PASS 3840x2160 x2')
