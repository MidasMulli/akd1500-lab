"""Host-resource tradeoff and frozen-forecast contention report from audited bytes."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coprocessor.service_archive import BASE,S,audit

LABELS={'native-spin':'AKD · spin','native-sleep':'AKD · sleep100µs','cpu-blas1':'CPU · OpenBLAS1','cpu-blas4':'CPU · OpenBLAS4','cpu-dense':'CPU · NEON dense','cpu-sparse':'CPU · zero-skipping','cpu-parallel':'CPU · OpenMP4','none':'Model only'}
COLORS={'native-spin':'#dcaa70','native-sleep':'#68d4c0','cpu-blas1':'#91abdc','cpu-blas4':'#bba1df','none':'#c3d1dc'}
def mode(r):return 'native-sleep' if r['mode']=='native' and r['receipt']['sleep_us'] else 'native-spin' if r['mode']=='native' else r['mode']
def stat(s):
    fields=s[s.rfind(')')+2:].split();return int(fields[11])+int(fields[12])
def cpu_ticks(s):return np.array(list(map(int,s.splitlines()[0].split()[1:9])),dtype='i8')

def results(report):
    rows=report['rows'];confirmation=[];busy=[]
    for r in rows:
        o=r['receipt']
        if o['repeat'] in [1,2] and o['side']==64 and o['contract']=='potentials' and o['batch']==1 and o['profile']==1 and o['count']==64:
            confirmation.append(dict(name=r['name'],mode=mode(r),seed=o['seed'],repeat=o['repeat'],arm=r['arm'],wall_us=r['amortized_wall_us_per_image'],process_cpu_us=r['amortized_process_cpu_us_per_image']))
    for r in report['windows']:
        if not r['busy']:continue
        o=r['receipt'];delta=cpu_ticks(o['host_stat_after'])-cpu_ticks(o['host_stat_before'])
        rec=dict(name=r['name'],mode=mode(r),seed=r['seed'],repeat=o['repeat'],foreground_fps=r['foreground_fps'],foreground_calls=r['foreground_calls_complete_in_window'],foreground_median_ms=r['foreground_median_ms'],foreground_p95_ms=r['foreground_p95_ms'],actor_operational_cpu_ms=r.get('total_actor_cpu_ms',0),actor_bracket_cpu_ms=(stat(o['actor_stat_after'])-stat(o['actor_stat_before']))*10 if o['actor_stat_before'] else 0,
                 host_busy_percent=float(100*(delta.sum()-delta[3]-delta[4])/delta.sum()),counter_bracket_ms=(o['host_stat_after_ns']-o['host_stat_before_ns'])/1e6,deadline_misses=r.get('missed_event_deadlines',0))
        busy.append(rec)
    assert len(busy)==15,'complete sealed three-block comparison required'
    summary={}
    for k in sorted(set(x['mode'] for x in confirmation)):
        z=[x for x in confirmation if x['mode']==k and x['arm']==1]
        summary[k]=dict(cells=len(z),wall_us=float(np.median([x['wall_us'] for x in z])),process_cpu_us=float(np.median([x['process_cpu_us'] for x in z])),wall_cell_range_us=[min(x['wall_us'] for x in z),max(x['wall_us'] for x in z)])
    gate=[]
    for rep in range(3):
        arm={x['mode']:x for x in busy if x['repeat']==rep};c=arm['native-sleep']
        gate.append(dict(repeat=rep,deadline_pass=c['deadline_misses']==0,against={key:dict(fps_ratio=c['foreground_fps']/arm[key]['foreground_fps'],p95_ratio=c['foreground_p95_ms']/arm[key]['foreground_p95_ms']) for key in ['cpu-blas1','cpu-blas4']}))
    practical=all(g['deadline_pass'] and all(v['fps_ratio']>=1.05 and v['p95_ratio']<=1.05 for v in g['against'].values()) for g in gate)
    return dict(status='AUDITED_MEASURED_RESOURCE_TRADEOFF',confirmation=confirmation,busy=busy,summary=summary,practical_gate_pass=practical,practical_gate_by_block=gate,physical_sessions=report['physical_sessions'],physical_forwards=report['physical_forwards'],failed_setups=report['preserved_failed_setups'],archive_sha256=report['archive_sha256'])

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def figure(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':'#e4edf4','axes.labelcolor':'#a6b8c9','xtick.color':'#a6b8c9','ytick.color':'#a6b8c9','axes.facecolor':'#101a27','figure.facecolor':'#080f19','axes.edgecolor':'#314053','savefig.facecolor':'#080f19'})
    fig=plt.figure(figsize=(19.2,10.8));gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=.72,wspace=.34,left=.11,right=.94,top=.77,bottom=.17)
    fig.text(.065,.925,'Native learning, with less host work.',fontsize=29,weight='bold')
    fig.text(.065,.87,'Exact outputs and memory retained. The CPU remains faster; foreground throughput is effectively tied.',fontsize=16,color='#b5c4d4')
    modes=['native-spin','native-sleep','cpu-blas4','cpu-blas1'];labels=[LABELS[k] for k in modes]
    for panel,(metric,title) in enumerate([('wall_us','Answer latency · ms/image'),('process_cpu_us','Process CPU during each call · ms/image')]):
        ax=fig.add_subplot(gs[0,panel]);ys=np.arange(len(modes))
        values=[data['summary'][k][metric]/1000 for k in modes];ax.barh(ys,values,color=[COLORS[k] for k in modes],height=.52)
        for i,k in enumerate(modes):
            zz=[x[metric]/1000 for x in data['confirmation'] if x['mode']==k and x['arm']==1];ax.scatter(zz,np.full(len(zz),i),s=15,color='#eef3f8',zorder=3)
            ax.text(values[i]+.12,i,f'{values[i]:.2f}',va='center',fontsize=13)
        ax.set(yticks=ys,yticklabels=labels,xlim=(0,9),title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.1);ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[1,:]);order=['none','cpu-blas4','cpu-blas1','native-spin','native-sleep'];y=np.arange(len(order))
    for i,k in enumerate(order):
        z=[x['foreground_fps'] for x in data['busy'] if x['mode']==k];ax.scatter(z,np.full(3,i),s=55,c=COLORS.get(k,'#91abdc'),zorder=3);ax.plot([min(z),max(z)],[i,i],color=COLORS.get(k,'#91abdc'),lw=2)
    ax.set(yticks=y,yticklabels=[LABELS[k] for k in order],xlabel='Frozen Nano forecasts completed per second · three rotated 3.2s windows per arm',xlim=(0,50),title='Busy Nano: no qualified ≥5% forecast-throughput advantage over both CPU controls');ax.invert_yaxis();ax.grid(axis='x',alpha=.1);ax.spines[['top','right']].set_visible(False)
    fig.text(.065,.075,'Three initialization seeds · two confirmation repetitions · 20Hz contention stream · zero skipped events',fontsize=12,color='#a6b8c9')
    fig.text(.065,.04,'Synthetic characterization + repeated frozen T202 forecasts. Process CPU is not energy. No electrical spike timing observed.',fontsize=12,color='#a6b8c9')
    fig.savefig(S/'RESULTS.png',dpi=200);fig.savefig(S/'RESULTS.svg');plt.close(fig)
    svg=S/'RESULTS.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

def main():
    report=audit();data=results(report);(S/'METRICS.json').write_text(json.dumps(data,indent=2)+'\n');write_csv(S/'CONFIRMATION.csv',data['confirmation']);write_csv(S/'BUSY_WINDOWS.csv',data['busy']);figure(data)
    print(json.dumps({k:v for k,v in data.items() if k not in ['confirmation','busy','summary']}))

if __name__=='__main__':main()
