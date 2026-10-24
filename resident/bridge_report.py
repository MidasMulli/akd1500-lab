"""Measured fixed-bridge outputs, controls and schematic composition."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from resident.bridge import BASE,P
from resident.bridge_audit import audit_public


def main():
    audit_public();meta=json.loads(BASE.with_suffix('.json').read_text())
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        actual=z['mixed_smoke__outputs'][0];expected=z['mixed_smoke__expected'][0]
    bg='#0a111b';ink='#edf2f8';muted='#99aec3';blue='#78c9ed';orange='#efb76c';line='#34485a'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':ink,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,'font.size':14})
    fig=plt.figure(figsize=(16,9),dpi=240,facecolor=bg)
    fig.text(.06,.932,'AKD1500 / ISOLATED NATIVE COMMUNICATION',size=15,color=muted)
    fig.text(.06,.86,'Fixed blocks pass. The mixed handoff fails.',size=29,weight='bold')
    fig.text(.06,.808,'5 owned sessions · 321 inference forwards · ordinary compiler programs · no learning or RAM patches',size=15,color=muted)
    a=fig.add_axes([.06,.53,.88,.23],facecolor=bg);a.set_axis_off()
    a.text(0,.92,'MATCHED NATIVE CONTROLS',size=14,color=muted,weight='bold')
    for j,(name,value,color) in enumerate([('FNP source → FNP comparison + encoder → FNP readout','128 / 128 exact',blue),
        ('Host input → CNP comparison + encoder → FNP readout','128 / 128 exact',blue),
        ('FNP source → CNP comparison + encoder → FNP readout','0 / 1 · stopped',orange)]):
        yy=.65-j*.25;a.text(0,yy,name,size=17);a.text(1,yy,value,size=18,color=color,ha='right',weight='bold')
    a=fig.add_axes([.09,.19,.36,.245],facecolor=bg);c=np.arange(4)
    a.bar(c-.18,expected,.32,color=blue,label='Expected');a.bar(c+.18,actual,.32,color=orange,label='Measured')
    a.set_ylim(0,154);a.set_yticks([0,64,128]);a.set_xticks(c,[f'Class {j}' for j in c]);a.set_ylabel('Readout count');a.set_title('Mixed graph / first input only',loc='left',color=ink,pad=18,weight='bold',fontsize=16)
    for s in ['top','right']:a.spines[s].set_visible(False)
    for s in ['left','bottom']:a.spines[s].set_color(line)
    a.legend(frameon=False,labelcolor=ink,loc='upper center',ncol=2,fontsize=12)
    a.text(-.18,132,'128',ha='center',color=blue);a.text(3.18,132,'128',ha='center',color=orange)
    a=fig.add_axes([.52,.19,.42,.245],facecolor=bg);a.set_axis_off()
    a.text(0,.99,'THE FAILED COMPOSITION / SCHEMATIC',size=13,color=muted,weight='bold')
    layers=meta['runs'][-1]['fixture']['mapping']
    for j,(label,color) in enumerate([('FNP\nsource',blue),('CNP\ncompare',orange),('CNP\nencoder',orange),('FNP\nreadout',blue)]):
        xx=j*.25
        a.text(xx+.10,.64,label,ha='center',va='center',size=14,color=color,bbox=dict(boxstyle='round,pad=.55',facecolor='#111e2c',edgecolor=line))
        placement=layers[j]['nps'][0]['np'];a.text(xx+.10,.30,str(tuple(placement)),ha='center',size=10,color=muted)
        if j<3:a.annotate('',xy=(xx+.23,.64),xytext=(xx+.20,.64),arrowprops=dict(arrowstyle='->',color=muted,lw=1.5))
    a.text(0,.06,'All captured fixed parameter RAM stayed unchanged.',size=12,color=muted)
    fig.text(.06,.10,'This failure exists without the learning adapter. Representation/routing at the handoff remains unresolved.',size=14)
    fig.text(.06,.06,'One counterexample stops expansion. Not a mixed-graph accuracy estimate, chip-capacity bound, learning result or CPU/energy claim.',size=11,color=muted)
    v=P/'visual';fig.savefig(v/'fixed-bridge-4k.png',facecolor=bg);plt.close(fig)
    files=[BASE.with_suffix('.json'),BASE.with_suffix('.npz'),Path(__file__),v/'fixed-bridge-4k.png']
    (v/'FIXED_BRIDGE_SOURCE_BINDING.json').write_text(json.dumps(dict(status='FIXED_OUTPUT_AND_PARAMETER_RAM_BOUND',
        hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')


if __name__=='__main__':main()
