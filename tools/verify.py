"""Independent Python integer/set reconstruction of saved native learning data."""
from pathlib import Path
import hashlib,json,math,struct
def read_words(p):
    b=p.read_bytes();assert len(b)%4==0;return list(struct.unpack('<'+'I'*(len(b)//4),b))
def verify(root,run):
    rec=json.loads((run/'RECEIPT.json').read_text());assert rec['status']=='PASS' and rec['device_released']
    for name,h in rec['hashes'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h,('raw hash changed',name)
    f=root/rec['fixture'];n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split());count=rec['samples'];stride=4+n//32
    initial=read_words(f.with_suffix('.ram'));state=initial[:];assert len(state)==2+u*stride
    source=f.with_suffix('.cpu_inputs') if f.with_suffix('.cpu_inputs').exists() else f.with_suffix('.inputs')
    raw=source.read_bytes();labels=read_words(f.with_suffix('.labels'))
    outputs=classes if rec.get('output_contract')==2 else u
    native=read_words(run/'native.scores');assert len(native)==outputs*count and native==read_words(run/'cpu.scores')
    weights=[]
    for j in range(u):
        base=2+j*stride;weights.append(sum(state[base+4+k]<<(32*k) for k in range(n//32)));assert weights[-1].bit_count()==nw
    rot=cursor=0;changed_records=set();changed_bits=0
    for sample in range(count):
        x=sum(int(bool(v))<<k for k,v in enumerate(raw[sample*n:(sample+1)*n]));scores=[(w&x).bit_count() for w in weights]
        expected=[int(v>rec['threshold']) for v in scores] if rec.get('activated') else scores
        if rec.get('output_contract')==2:expected=[sum(expected[k*(u//classes):(k+1)*(u//classes)]) for k in range(classes)]
        assert expected==native[sample*outputs:(sample+1)*outputs],('independent output mismatch',sample)
        if rec['learning']:
            mix=lambda p:((p>>2)&7)|((p&3)<<3)
            rot=(rot+sum(mix(p) for p in scores))&31;lo=labels[sample]*(u//classes);hi=lo+u//classes
            eligible=[j for j in range(lo,hi) if scores[j]>max(0,state[2+j*stride+3]>>16)]
            if eligible:
                win=max(eligible,key=lambda j:(scores[j],-j));base=2+win*stride;before=weights[win]
                remove={k for k in range(n) if before>>k&1 and not x>>k&1};insert={k for k in range(n) if x>>k&1 and not before>>k&1}
                k=min(state[base+3]&65535,len(remove),len(insert));r=[];a=[];word=cursor;burst=0;scanned=0
                if k:
                    while len(r)<k or len(a)<k:
                        for pool,chosen in [(remove,r),(insert,a)]:
                            possible=[z for z in pool if z//32==word]
                            if len(chosen)<k and possible:
                                z=min(possible,key=lambda z:((z%32)-rot)%32);chosen.append(z);pool.remove(z)
                        rot=(rot+mix(scores[-1]))&31;scanned+=1
                        if len(r)==k and len(a)==k:break
                        word=(word+1)%(n//32);burst=0 if word==0 else (burst+1)%4
                        assert scanned<=(n//32)*(k+1)
                    cursor=(word+1)%(n//32);rot=(rot+min(3-burst,n//32-1-word)*mix(scores[-1]))&31
                    for z in r:weights[win]&=~(1<<z)
                    for z in a:weights[win]|=1<<z
                    oldswap=state[base+3]&65535;swap=max(math.ceil(nw*.25),oldswap-math.ceil(math.floor(nw*.5+.5)*.125));threshold=min(scores[win],math.floor(nw*.75+.5))
                    state[base+3]=(threshold<<16)|swap
                    for z in range(n//32):state[base+4+z]=(weights[win]>>(32*z))&0xffffffff
                    changed_records.add(win);changed_bits+=(before^weights[win]).bit_count()
        snapshot=run/f'native.snapshot_{sample}.ram'
        if snapshot.exists():assert state==read_words(snapshot),('per-step measured RAM mismatch',sample)
    assert state==read_words(run/'native.ram')==read_words(run/'cpu.ram'),'complete final RAM mismatch'
    return dict(status='PASS',run=run.name,inputs=n,neurons=u,samples=count,learning=bool(rec['learning']),changed_neuron_records=len(changed_records),connection_bit_edits=changed_bits,exact_score_values=count*outputs,fixture=rec['fixture'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1];results=[]
    for run in sorted((a.study/'raw').iterdir()):
        if (run/'RECEIPT.json').exists():
            r=json.loads((run/'RECEIPT.json').read_text())
            if r.get('status')=='PASS':results.append(verify(root,run))
    z=dict(status='PASS',runs=len(results),results=results);(a.study/'evidence/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps({k:v for k,v in z.items() if k!='results'}))
