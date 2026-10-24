"""Scientific plots from recounted native counters; no simulated spike timing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from metal.audit_public import audit
from metal.ablation import recount

P = Path(__file__).resolve().parents[1]
BASE = P / 'results/2026-10-23-metal'
BG, PANEL, INK, MUTED = '#08111e', '#101e30', '#edf3fa', '#a6b6ca'
CYAN, GOLD = '#74e3cf', '#ffbd78'


def ablation_figure():
    base=P/'results/2026-10-23-metal-ablation'
    r=json.loads(base.with_suffix('.json').read_text())
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    recount(r,a)
    index=int(np.flatnonzero(a['baseline__inputs'].sum(axis=1)==512)[0])
    names=['baseline','suppress_A','suppress_B','suppress_both']
    values=np.asarray([a[name+'__outputs'][index] for name in names])
    fig=plt.figure(figsize=(16,9),dpi=200,facecolor=BG)
    fig.text(.085,.91,'Which physical NPU contributes?',fontsize=26,weight='bold')
    fig.text(.085,.85,'Selective threshold suppression, with actual NPU memory readbacks before and after every run',fontsize=12,color=MUTED)
    ax=fig.add_axes([.12,.29,.79,.46],facecolor=PANEL)
    x=np.arange(4);ax.bar(x-.17,values[:,0],width=.32,color=CYAN,label='Group A · CNP1 (1,2,1)')
    ax.bar(x+.17,values[:,1],width=.32,color=GOLD,label='Group B · CNP1 (1,2,2)')
    for i,row in enumerate(values):
        for offset,v in zip([-.17,.17],row):ax.text(i+offset,int(v)+4,str(int(v)),ha='center',fontsize=13)
    ax.set(ylim=(0,165),xticks=x,xticklabels=['Baseline','Suppress A','Suppress B','Suppress both'],ylabel='Native group-support count')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#2d4058',alpha=.6);ax.set_axisbelow(True)
    ax.legend(loc='upper right',frameon=False,labelcolor=INK,fontsize=11)
    fig.text(.085,.18,'Weights, input stream and readout stay fixed. Only the selected NPU’s 128 thresholds change from 15 to 32.',fontsize=12,color=INK)
    fig.text(.085,.125,'Displayed: a fully active 512-bit input. All 64 inputs in each of four treatments match independent numerical references.',fontsize=11,color=MUTED)
    fig.text(.085,.08,'Four physical sessions · 256 native forwards · 40 region readbacks · contribution test, not a per-cycle simultaneity claim',fontsize=11,color=MUTED)
    png=P/'visual/native-npu-ablation.png';svg=P/'visual/native-npu-ablation.svg'
    fig.savefig(png,facecolor=BG,dpi=200);fig.savefig(svg,facecolor=BG);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    return base,png,svg


def main():
    audit()
    report = json.loads(BASE.with_suffix('.json').read_text())
    by_name = {r['name']:r for r in report['runs']}
    def metric(name, density):
        g = next(g for g in by_name[name]['audit']['groups']
                 if g['phase'] == 'initial_inference' and g['density'] == density)
        return g['median_cycles']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
        'text.color':INK, 'axes.labelcolor':MUTED, 'xtick.color':MUTED,
        'ytick.color':MUTED, 'axes.edgecolor':'#3c5068'})
    fig = plt.figure(figsize=(19.2,10.8), dpi=200, facecolor=BG)
    axes = [fig.add_axes(rect, facecolor=PANEL) for rect in
            [[.065,.54,.385,.285],[.565,.54,.365,.285], [.065,.165,.385,.27],[.565,.165,.365,.27]]]
    for ax in axes[:3]:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#2d4058', alpha=.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
    fig.text(.065,.94,'AKD1500 / execution under test',fontsize=29,weight='bold')
    fig.text(.065,.895,'24 native sessions  ·  9,568 forwards  ·  4,208 complete learning-state snapshots  ·  one physical chip',fontsize=13,color=MUTED)
    titles=['01  More FNP3 records add service work','02  More NPUs provide usable capacity',
            '03  Activity sensitivity depends on the path','04  Teaching bursts passed both settings']
    for ax,title in zip(axes,titles): ax.set_title(title,loc='left',pad=18,fontsize=15,weight='bold')
    a=axes[0]; widths=[8,32,64,128,256,512]
    a.plot(widths,[metric(f'screen_width{u}',256) for u in widths],'-o',color=CYAN,lw=2.4,label='Raw neuron scores')
    a.plot([64,512],[metric(f'control_compact{u}',256) for u in [64,512]],'--s',color=GOLD,lw=2,label='Two counts + added readout NPU')
    a.set(xlim=(0,540),ylim=(0,19000),xticks=[8,128,256,512],xlabel='Configured neuron records in one FNP3 NPU',ylabel='DMA monitor cycles / input')
    a.legend(loc='upper left',frameon=False,labelcolor=INK,fontsize=10)
    a.text(280,3900,'~20 added monitor cycles\nper added record in this path',color=CYAN,fontsize=11)
    b=axes[1]; vals=[metric(f'screen_cnp{u}',256) for u in [64,128,256,512]]
    b.bar(np.arange(4),vals,color=['#4c7893',CYAN,GOLD,'#4c7893'],width=.58)
    for i,v in enumerate(vals): b.text(i,v+650,f'{v:,.0f}',ha='center',fontsize=11,color=INK)
    b.set(ylim=(0,27000),xticks=range(4),xticklabels=['64 / 1 NP','128 / 1 NP','256 / 2 NPs','512 / 3 NPs'],
          xlabel='Convolution outputs / compute NPUs (+ one fixed readout NPU)',ylabel='DMA monitor cycles / input')
    b.text(.03,.91,'128 → 256 outputs: +0.065% monitor cost',transform=b.transAxes,fontsize=11,color=GOLD)
    c=axes[2]; densities=[0,1,8,32,128,256,512]
    c.plot(densities,[metric('screen_width64',d) for d in densities],'-o',color=CYAN,lw=2,label='FNP3 raw-score path, 64 records')
    c.plot(densities,[metric('screen_cnp64',d) for d in densities],'--s',color=GOLD,lw=2,label='CNP fixed path, 64 outputs')
    c.set_xscale('symlog',linthresh=1)
    c.set(xlim=(0,650),ylim=(0,20000),xticks=densities,xticklabels=list(map(str,densities)),
          xlabel='Active input bits out of 512; dense host transport',ylabel='DMA monitor cycles / input')
    c.legend(loc='upper left',frameon=False,labelcolor=INK,fontsize=10)
    c.text(.02,.04,'Different kernels/output contracts; this is within-path sensitivity.',transform=c.transAxes,fontsize=9,color=MUTED)
    d=axes[3]; d.set_axis_off()
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        key='teaching_queue_pipeline0__'
        initial=z[key+'initial'][2:].reshape(64,20)
        final=z[key+'final'][2:].reshape(64,20)
        changed=(initial!=final).any(axis=1)
    for k,(name,setting) in enumerate([('Normal learning setting','OFF'),('Known pipeline flag enabled','ON')]):
        x0=.05+k*.51
        d.text(x0,.88,setting,fontsize=20,weight='bold',color=CYAN if k==0 else GOLD,transform=d.transAxes)
        for j in range(64):
            x=x0+(j%8)*.035; y=.78-(j//8)*.035
            color=(CYAN if k==0 else GOLD) if changed[j] else '#334962'
            d.scatter([x],[y],s=28,marker='s',color=color,transform=d.transAxes)
        d.text(x0,.30,'4,096 / 4,096 outputs exact\n8 / 8 RAM boundaries exact',fontsize=11,transform=d.transAxes,linespacing=1.6)
        d.text(x0,.15,name,fontsize=9,color=MUTED,transform=d.transAxes)
    d.text(.05,.02,'Squares: measured records changed after 64 teaching inputs.\nSame input/label stream; this does not prove overlapping updates.',fontsize=10,color=MUTED,transform=d.transAxes,linespacing=1.6)
    fig.text(.065,.075,'Counter scope: DMA-path monitoring, not arithmetic-only cycles or per-neuron timing. No CPU or electrical-energy advantage claimed.',fontsize=11,color=MUTED)
    fig.text(.065,.04,'Source: MidasMulli/akd1500-lab  ·  results/2026-10-23-metal.{json,npz}  ·  exact numerical recount and preserved compiler rejection',fontsize=10,color=MUTED)
    png=P/'visual/native-execution-4k.png'; svg=P/'visual/native-execution.svg'
    fig.savefig(png,facecolor=BG,dpi=200);fig.savefig(svg,facecolor=BG);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    ablation_base,ablation_png,ablation_svg=ablation_figure()
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (P/'visual/NATIVE_EXECUTION_SOURCE_BINDING.json').write_text(json.dumps(dict(
        report_sha256=h(BASE.with_suffix('.json')),archive_sha256=h(BASE.with_suffix('.npz')),
        generator_sha256=h(Path(__file__)),png_sha256=h(png),svg_sha256=h(svg),
        ablation_report_sha256=h(ablation_base.with_suffix('.json')),ablation_archive_sha256=h(ablation_base.with_suffix('.npz')),
        ablation_png_sha256=h(ablation_png),ablation_svg_sha256=h(ablation_svg),
        reconstruction='Scientific plots of native counters and RAM record edits; no electrical spike animation'),indent=2)+'\n')
    print(png)


if __name__=='__main__':main()
