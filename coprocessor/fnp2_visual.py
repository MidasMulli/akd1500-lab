"""Recorded address/control evidence; no internal electrical trace claim."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from coprocessor.fnp2_archive import BASE,audit_public,P
def main():
    audit_public();meta=json.loads(BASE.with_suffix('.json').read_text())
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        before=z['boundary_above__initial'];after=z['boundary_native_above_l1__final'];guard=z['boundary_native_above_l1__guard_counts']
        words=np.flatnonzero(before!=after)
        old=before[words];new=after[words]
    bg='#0a101a';fg='#e6edf6';muted='#9dafc4';line='#2b3d52';blue='#78baff';green='#82c7aa';red='#ee9a90'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':fg,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':fg,'font.size':12,'savefig.facecolor':bg})
    fig=plt.figure(figsize=(19.2,10.8),dpi=200,facecolor=bg)
    fig.text(.055,.94,'Inference capacity is not learning coverage',fontsize=30,weight='bold')
    fig.text(.055,.897,'AKD1500 FNP2  /  same 8,192-record bank  /  matched CPU and frozen controls  /  recorded results',fontsize=14,color=muted)
    items=[('INFERENCE','Both targets pass',blue),('TARGET 6500','Correct native update',green),('TARGET 6600','Off-target update',red),('HOST RECORD WRITES',str(int(guard[0])),fg)]
    for i,(tag,value,color) in enumerate(items):
        x=.055+i*.235;fig.add_artist(FancyBboxPatch((x,.747),.20,.106,boxstyle='round,pad=.006,rounding_size=.008',transform=fig.transFigure,facecolor='#111d2c',edgecolor=line))
        fig.text(x+.012,.822,tag,fontsize=10.5,color=color,weight='bold');fig.text(x+.012,.779,value,fontsize=17,weight='bold')
    fig.text(.055,.692,'A unique class-1 winner scores 31. The potential is correct; the stored update is not.',fontsize=17,weight='bold')
    ax=fig.add_axes([.16,.351,.775,.279]);ax.set_facecolor(bg);ax.set_xlim(-12,655);ax.set_ylim(-.55,1.7)
    lo=(6500*20+3)*4/1024;hi=(6600*20+3)*4/1024;alias=931*4/1024
    ax.axvspan(0,640,color='#122136');ax.axvline(512,color=muted,ls='--',lw=1.3)
    ax.plot([lo],[1],marker='o',color=green,ms=11);ax.text(lo-12,1.28,'6500 → 6500',ha='right',fontsize=15,color=green)
    ax.plot([hi],[0],marker='o',color=blue,ms=11);ax.text(hi+10,.21,'Expected record 6600',fontsize=14,color=blue)
    ax.annotate('',xy=(alias,0),xytext=(hi,0),arrowprops=dict(arrowstyle='->',lw=2.5,color=red));ax.plot([alias],[0],marker='s',ms=10,color=red)
    ax.text(alias+10,-.27,'Observed metadata in record 46',fontsize=14,color=red)
    ax.text(512,1.5,'512 KiB offset',ha='center',fontsize=11,color=muted)
    ax.set_yticks([0,1],['Higher-address probe','Lower-address control']);ax.set_xticks([0,128,256,384,512,640]);ax.set_xlabel('Record-buffer offset (KiB) · buffer registered at 0x20000140',labelpad=13)
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(line);ax.tick_params(axis='y',length=0,pad=15);ax.grid(axis='x',color=line,alpha=.6)
    fig.text(.055,.264,'Exact changed words in the actual record buffer',fontsize=17,weight='bold')
    for i,(word,a,b) in enumerate(zip(words,old,new)):
        x=.055+i*.24;fig.text(x,.222,f'Word {int(word)}',fontsize=14,color=muted);fig.text(x,.184,f'{int(a):08x}  →  {int(b):08x}',fontsize=17,family='DejaVu Sans Mono',color=red)
    fig.text(.795,.235,'EXPANSION HALTED',fontsize=13,weight='bold',color=red);fig.text(.795,.203,'Full RAM + failed prefix saved',fontsize=11,color=muted)
    fig.text(.055,.126,'Expected metadata word 132003 appears at word 931: a 524,288-byte separation. Internal cause remains unresolved.',fontsize=13)
    fig.text(.055,.087,'The observer guards writes through our transport; it is not hardware access control. No silicon erratum, maximum-neuron or energy claim.',fontsize=11,color=muted)
    fig.text(.055,.055,'Controlled cases: same program/weights, one active bit replaced, unique winner. Two-forward failing prefix; complete native state independently recounted.',fontsize=11,color=muted)
    out=P/'visual/fnp2-address-boundary-4k';fig.savefig(out.with_suffix('.png'));fig.savefig(out.with_suffix('.svg'));plt.close(fig)
    svg=out.with_suffix('.svg');svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    binding=dict(status='RECORDED_OBSERVATION_NOT_INTERNAL_WIRE_TRACE',numeric_archive_sha256=sha(BASE.with_suffix('.npz')),metadata_sha256=sha(BASE.with_suffix('.json')),renderer_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in [out.with_suffix('.png'),out.with_suffix('.svg')]})
    (P/'visual/FNP2_ADDRESS_SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
if __name__=='__main__':main()
