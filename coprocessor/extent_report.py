"""Figures/CSV computed from independently audited recorded arrays."""
import csv,json,re
from pathlib import Path
import numpy as np
from coprocessor.extent_audit import BASE,S,audit,masks

def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    checked=audit();meta=json.loads(BASE.with_suffix('.json').read_text());records=[];groups={}
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for name,run in meta['runs'].items():
            match=re.fullmatch(r'(.*)_s(\d+)_s(\d+)_a([01])_n64_r([012])',name)
            if not match:continue
            mode,side,seed,arm,rep=match.groups();side,seed,arm,rep=map(int,[side,seed,arm,rep]);r=run['receipt']
            t=z[run['arrays']['latency_ns']['key']].reshape(r['repeats'],64)[:,8:]/1e6
            row=dict(run=name,mode=mode,side=side,seed=seed,learning=arm,repeat=rep,timing_endpoint=rep in [1,2],median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),events=int(t.size))
            records.append(row)
            if arm==1 and rep in [1,2]:groups.setdefault((side,mode),[]).append(t.ravel())
        summary=[]
        for (side,mode),values in sorted(groups.items()):
            t=np.concatenate(values);summary.append(dict(side=side,mode=mode,median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),timed_events=len(t),fresh_seeds=1 if side==32 else 3,isolated_process_repeats_per_seed=2,internal_stream_repeats=1 if mode=='native' else 3))
        with (S/'TIMINGS.csv').open('w') as out:
            w=csv.DictWriter(out,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
        result=dict(archive_sha256=meta['archive_sha256'],endpoint='Learning arm, events9..64, isolated rotated timing repetitions1/2 only',groups=summary,physical_audit={k:v for k,v in checked.items() if k!='rows'})
        (S/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
        bg='#070c14';ink='#e7f0f8';muted='#8ba1b5';cyan='#66cfed';green='#70e7b5';amber='#edb168'
        plt.rcParams.update({'figure.facecolor':bg,'axes.facecolor':bg,'axes.edgecolor':'#273a4c','text.color':ink,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none'})
        fig=plt.figure(figsize=(16,9),dpi=240)
        fig.text(.055,.94,'Larger native workloads, stronger CPU controls',fontsize=24,weight='bold')
        fig.text(.055,.893,'4× image work qualified · exact native connection changes · CPU still faster',fontsize=15,color=muted)
        ax=fig.add_axes([.075,.29,.47,.51]);colors={'native':cyan,'cpu-blas4':green,'cpu-blas1':'#9aa8ef','cpu-parallel':amber,'cpu-sparse':'#bc8fab','cpu-dense':'#7c8b9c'}
        labels={'native':'AKD · full input → output','cpu-blas4':'CPU · OpenBLAS, 4 threads','cpu-blas1':'CPU · OpenBLAS, 1 thread','cpu-parallel':'CPU · NEON/OpenMP, 4 threads','cpu-sparse':'CPU · NEON, skip zeros','cpu-dense':'CPU · NEON dense'}
        for mode in ['cpu-dense','cpu-sparse','cpu-blas1','native','cpu-parallel','cpu-blas4']:
            g=sorted([r for r in summary if r['mode']==mode],key=lambda r:r['side']);ax.plot([r['side'] for r in g],[r['median_ms'] for r in g],marker='o',lw=2.5 if mode in ['native','cpu-blas4'] else 1.2,color=colors[mode],label=labels[mode])
        ax.set_yscale('log');ax.set_xticks([32,64],['32×32 RGB\n9.72M nominal MACs','64×64 RGB\n38.90M nominal MACs']);ax.set_ylabel('Median operational latency (ms, log scale)');ax.grid(axis='y',alpha=.15);ax.set_xlim(27,69);ax.legend(loc='upper left',fontsize=10,frameon=False)
        ax.spines[['top','right']].set_visible(False)
        for mode in ['native','cpu-blas4']:
            for r in [v for v in summary if v['mode']==mode]:ax.annotate(f'{r["median_ms"]:.2f} ms',(r['side'],r['median_ms']),xytext=(9,1),textcoords='offset points',color=colors[mode],fontsize=12,weight='bold')
        bx=fig.add_axes([.64,.34,.31,.44]);cell='s64_s107501';f=meta['fixtures'][cell];initial=z[f['arrays']['ram']['key']]
        rn=meta['runs'][f'native_{cell}_a1_n64_r0'];states=z[rn['arrays']['snapshots']['key']].reshape(64,len(initial));before=np.concatenate([initial[None],states[:-1]])
        bm,am=masks(before,512),masks(states,512);added=(am>bm).sum(axis=(1,2))
        bx.axvspan(8.5,40.5,color=cyan,alpha=.07);bx.vlines(np.arange(1,65),0,added,color=green,lw=2);bx.scatter(np.arange(1,65),added,color=green,s=12)
        bx.set_title('Actual on-chip connection edits',loc='left',fontsize=16,weight='bold',pad=18);bx.set_xlabel('Recorded event');bx.set_ylabel('Added connections; equal removals');bx.set_xlim(0,65);bx.set_ylim(-.5,max(added)+4);bx.spines[['top','right']].set_visible(False);bx.grid(axis='y',alpha=.15)
        bx.text(22, max(added)+2,'Teaching enabled',ha='center',fontsize=10,color=cyan);bx.text(51,max(added)+1,'Queries\n0 edits',ha='center',fontsize=11,color=muted)
        fig.text(.64,.25,'64×64: 3 fresh seeds · full learner RAM exact each event',fontsize=11,color=muted)
        fig.text(.055,.17,'REFERENCE PASSED',fontsize=10,color=cyan,weight='bold');fig.text(.055,.125,'Vendor VWW · 28 NPUs + HRC\nSingle pass · every output matched',fontsize=12)
        fig.text(.39,.17,'LEARNING RESIDENCY',fontsize=10,color=cyan,weight='bold');fig.text(.39,.125,'3 CNPs + 1 FNP3 + HRC\n10,248-byte learner packet verified',fontsize=12)
        fig.text(.72,.17,'BOUNDARIES PRESERVED',fontsize=10,color=amber,weight='bold');fig.text(.72,.125,'1,024 channels / 96×96 compiler gates\nPower rig awaiting parts + calibration',fontsize=12)
        fig.text(.055,.055,'Synthetic inputs/labels · CPU input packing + exact learning included · native transfer included · diagnostics excluded · no energy claim',fontsize=10,color=muted)
        fig.savefig(S/'RESULTS.png',dpi=240);fig.savefig(S/'RESULTS.svg');plt.close(fig)
        svg=S/'RESULTS.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
