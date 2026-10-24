"""4K scientific figure from the complete recounted native archive."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from kernel.report import load_summary

P = Path(__file__).resolve().parents[1]
BG, PANEL, INK, MUTED = '#09131f', '#101f30', '#edf4fc', '#a6b8cc'
BLUE, GOLD, GREEN = '#72caff', '#ffc17d', '#8edbb9'


def main():
    report = load_summary()
    widths = {w['detector_records']: w for w in report['widths']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#3b526b'})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
    fig.text(.06, .946, 'AKD1500 / a native graph rewrite, measured', fontsize=28, weight='bold')
    fig.text(.06, .904, '16 physical sessions  ·  2,624 forwards  ·  exact outputs + fixed parameter RAM  ·  one chip', fontsize=13, color=MUTED)
    rects = [[.065, .555, .385, .245], [.565, .555, .36, .245], [.075, .18, .375, .235], [.565, .18, .36, .235]]
    a, b, c, d = [fig.add_axes(rect, facecolor=PANEL) for rect in rects]
    a.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
    for spine in a.spines.values(): spine.set_visible(False)
    a.set_title('01  Compact inside the chip; keep external I/O', loc='left', pad=22, weight='bold', fontsize=14)
    def node(x, y, width, title, detail, color):
        a.add_patch(FancyBboxPatch((x, y), width, .20, boxstyle='round,pad=.009,rounding_size=.025',
            edgecolor=color, facecolor=BG, linewidth=1.1))
        a.text(x + width/2, y + .13, title, ha='center', va='center', fontsize=10.5, color=color, weight='bold')
        a.text(x + width/2, y + .055, detail, ha='center', va='center', fontsize=8.5, color=MUTED)
    def arrow(x, y, end):
        a.annotate('', (end, y), (x, y), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.1))
    a.text(.025, .89, 'Direct · 2 NPUs', color=BLUE, fontsize=11, weight='bold')
    node(.035, .60, .20, '512 inputs', '4 identical copies', BLUE)
    node(.365, .60, .28, '512 detectors', '512-input records', BLUE)
    node(.785, .60, .18, '2 counts', 'fixed readout', BLUE)
    arrow(.245, .70, .35); arrow(.655, .70, .77)
    a.text(.025, .48, 'Rewrite · 3 NPUs', color=GOLD, fontsize=11, weight='bold')
    node(.035, .21, .20, '512 inputs', 'same host bytes', GOLD)
    node(.295, .21, .18, '128 bits', 'native encoder', GOLD)
    node(.535, .21, .22, '512 detectors', '128-input records', GOLD)
    node(.815, .21, .15, '2 counts', 'same output', GOLD)
    arrow(.245, .31, .28); arrow(.485, .31, .52); arrow(.765, .31, .80)
    a.text(.025, .06, 'Captured parameter payload: 41,120 → 26,784 bytes (−34.9%)', fontsize=10.5, color=GREEN)
    for ax in [b, c, d]:
        ax.spines[['top', 'right']].set_visible(False); ax.set_axisbelow(True)
        ax.grid(axis='y', color='#30455b', alpha=.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k'))
    b.set_title('02  Benefit depends on activity', loc='left', pad=22, weight='bold', fontsize=14)
    serial = widths[512]['paired_sessions'][0]['serial']
    for key, label, color, marker in [('direct_cycles','Direct',BLUE,'o'), ('compacted_cycles','Rewrite',GOLD,'s')]:
        vals = [r[key] for r in serial]
        b.plot([0, 64, 128], vals, marker=marker, lw=2.3, color=color, label=label)
        for x, y in zip([0, 64, 128], vals):
            b.annotate(f'{int(y):,}', (x,y), (0, 10 if key=='direct_cycles' else -19), textcoords='offset points', ha='center', fontsize=10)
    b.set(xlim=(-8, 136), ylim=(14500, 18900), xticks=[0,64,128], xticklabels=['0%','50%','100%'],
        xlabel='Active latent inputs · 512 detector records', ylabel='Median DMA service cycles / input')
    b.legend(loc='upper left', frameon=False, labelcolor=INK, fontsize=10)
    c.set_title('03  The extra NPU has a crossover cost', loc='left', pad=22, weight='bold', fontsize=14)
    x = np.arange(2)
    for i, (key,label,color,offset) in enumerate([('direct_cycles','Direct',BLUE,-.18),('compacted_cycles','Rewrite',GOLD,.18)]):
        vals = [widths[u]['paired_sessions'][0]['serial'][1][key] for u in [256,512]]
        c.bar(x+offset, vals, width=.32, color=color, label=label, zorder=3)
        for xx, y in zip(x+offset,vals): c.text(xx,y+380,f'{int(y):,}',ha='center',fontsize=10)
    c.set(ylim=(0,24000), xticks=x, xticklabels=['256 records','512 records'],
        xlabel='Half-active inputs · 3 fresh-program repeats per cell', ylabel='Median DMA service cycles / input')
    c.legend(loc='upper left', frameon=False, labelcolor=INK, fontsize=10, ncols=2)
    c.text(.07,.82,'18.2% more service',transform=c.transAxes,fontsize=11,color=GOLD)
    c.text(.58,.82,'5.9% less service',transform=c.transAxes,fontsize=11,color=GREEN)
    d.set_title('04  Queued zero-activity suffix: 21.1% less service', loc='left', pad=22, weight='bold', fontsize=14)
    pair = widths[512]['paired_sessions'][0]['burst']
    vals = [pair['direct_cycles'][0],pair['compacted_cycles'][0]]
    d.bar([0,1],vals,width=.5,color=[BLUE,GOLD],zorder=3)
    for xx,y in enumerate(vals): d.text(xx,y+3800,f'{y:,}',ha='center',fontsize=12)
    d.set(ylim=(0,180000),xticks=[0,1],xticklabels=['Direct','Rewrite'],
        xlabel='8 queued zero-input jobs · 512 detector records', ylabel='DMA service cycles / 8-job burst')
    d.text(.04,.96,'9 burst pairs · identical counts\nPreload and result drainage excluded',transform=d.transAxes,va='top',fontsize=10,color=MUTED)
    fig.text(.06,.085,'A bounded native graph improvement. The learning-policy candidates failed quality gates and were not run on the chip.',fontsize=12)
    fig.text(.06,.045,'Synthetic four-copy input domain; extra encoder NPU. DMA service ≠ arithmetic-only cycles. No CPU or calibrated energy advantage established.',fontsize=10,color=MUTED)
    outputs = ['visual/native-compaction-4k.png','visual/native-compaction.svg','visual/native-compaction-data.csv']
    fig.savefig(P/outputs[0],dpi=200,facecolor=BG);fig.savefig(P/outputs[1],facecolor=BG);plt.close(fig)
    svg=P/outputs[1];svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with (P/outputs[2]).open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['records','stage','latent_active_bits','direct_median_DMA_cycles','rewrite_median_DMA_cycles','reduction_percent','direct_host_median_ns','rewrite_host_median_ns'])
        for u,w in widths.items():
            for pair in w['paired_sessions']:
                for row in pair['serial']:
                    writer.writerow([u,pair['stage'],row['latent_active_bits'],row['direct_cycles'],row['compacted_cycles'],row['reduction_percent'],row['direct_host_median_ns'],row['compacted_host_median_ns']])
    paths=['results/2026-10-23-native-compaction.json','results/2026-10-23-native-compaction.npz',
        'results/2026-10-23-compaction-assessment.json','kernel/design.py','kernel/experiment.py','kernel/report.py',
        'tools/visualize_native_compaction.py']+outputs
    (P/'visual/NATIVE_COMPACTION_SOURCE_BINDING.json').write_text(json.dumps(dict(status='BOUND_TO_COMPLETE_NATIVE_RECOUNT',
        hashes={p:hashlib.sha256((P/p).read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    print(P/outputs[0])


if __name__=='__main__':main()
