"""Independent integer/set reconstruction; intermediate responses are derived."""
from pathlib import Path
import hashlib,json,math
from verify import read_words

class Learner:
    def __init__(self,ram,n=512,u=64):
        self.state=ram[:];self.n=n;self.u=u;self.stride=4+n//32;self.rot=0;self.cursor=0;self.edits=0;self.records=set()
        self.weights=[sum(ram[2+j*self.stride+4+k]<<(32*k) for k in range(n//32)) for j in range(u)]
        assert all(w.bit_count()==32 for w in self.weights)
    def forward(self,x,label,learn):
        n=self.n;stride=self.stride;scores=[(w&x).bit_count() for w in self.weights]
        if not learn:return scores
        mix=lambda p:((p>>2)&7)|((p&3)<<3)
        self.rot=(self.rot+sum(map(mix,scores)))&31
        lo=label*(self.u//2);hi=lo+self.u//2
        eligible=[j for j in range(lo,hi) if scores[j]>max(0,self.state[2+j*stride+3]>>16)]
        if not eligible:return scores
        win=max(eligible,key=lambda j:(scores[j],-j));base=2+win*stride;before=self.weights[win]
        remove={k for k in range(n) if before>>k&1 and not x>>k&1};insert={k for k in range(n) if x>>k&1 and not before>>k&1}
        count=min(self.state[base+3]&65535,len(remove),len(insert));r=[];a=[];word=self.cursor;burst=0;scanned=0
        if not count:return scores
        while len(r)<count or len(a)<count:
            for pool,chosen in [(remove,r),(insert,a)]:
                possible=[z for z in pool if z//32==word]
                if len(chosen)<count and possible:
                    z=min(possible,key=lambda z:((z%32)-self.rot)%32);chosen.append(z);pool.remove(z)
            self.rot=(self.rot+mix(scores[-1]))&31;scanned+=1
            if len(r)==count and len(a)==count:break
            word=(word+1)%(n//32);burst=0 if word==0 else (burst+1)%4
            assert scanned<=(n//32)*(count+1)
        self.cursor=(word+1)%(n//32);self.rot=(self.rot+min(3-burst,n//32-1-word)*mix(scores[-1]))&31
        for z in r:self.weights[win]&=~(1<<z)
        for z in a:self.weights[win]|=1<<z
        swap=max(8,(self.state[base+3]&65535)-2);threshold=min(scores[win],24);self.state[base+3]=(threshold<<16)|swap
        for k in range(n//32):self.state[base+4+k]=(self.weights[win]>>(32*k))&0xffffffff
        self.records.add(win);self.edits+=(before^self.weights[win]).bit_count()
        return scores

def verify(root,run):
    rec=json.loads((run/'RECEIPT.json').read_text());assert rec['status']=='PASS' and rec['device_released']
    for name,h in rec['hashes'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h,('raw hash changed',name)
    f=root/rec['fixture'];plan=json.loads(f.with_suffix('.json').read_text());nb=len(plan['banks']);mask=rec['learning_mask'];count=rec['samples']
    for name,h in rec['fixture_hashes'].items():assert hashlib.sha256((f.parent/name).read_bytes()).hexdigest()==h,('fixture hash changed',name)
    assert plan['inputs']==512 and plan['neurons']==64 and plan['threshold']==15 and plan['classes']==2 and plan['weights']==32
    assert plan.get('encoder','positive_8_copy') in ['positive_8_copy','dual_rail']
    assert 0<=mask<(1<<nb) and 0<count<=256
    banks=[Learner(read_words(f.parent/(f.name+(f'.compiler-bank{k}.ram' if rec.get('compiler_inference') else f'.bank{k}.ram')))) for k in range(nb)]
    for k,b in enumerate(banks):assert b.state==read_words(run/f'native.bank{k}.initial.ram')
    inputs=f.with_suffix('.inputs').read_bytes();labels=read_words(f.with_suffix('.labels'));native=read_words(run/'native.scores');assert len(native)==2*count
    responses=[]
    for i in range(count):
        x=sum(int(bool(v))<<k for k,v in enumerate(inputs[i*512:(i+1)*512]));step=[]
        for k,b in enumerate(banks):
            p=b.forward(x,labels[i],bool(mask&(1<<k)));r=[int(v>15) for v in p];step.append(r)
            assert b.state==read_words(run/f'native.snapshot_{i}_bank{k}.ram'),('bank snapshot mismatch',i,k)
            x=sum(int(v if plan.get("encoder")!="dual_rail" or bit<4 else not v)<<(j*8+bit) for j,v in enumerate(r) for bit in range(8))
        assert [sum(r[:32]),sum(r[32:])]==native[2*i:2*i+2],('final output mismatch',i)
        responses.append(step)
    for k,b in enumerate(banks):assert b.state==read_words(run/f'native.bank{k}.ram')
    for j in range(len(plan['fixed'])):
        a=read_words(run/f'native.fixed{j}.initial.ram');assert a==read_words(run/f'native.fixed{j}.final.ram')==read_words(f.parent/(f.name+f'.fixed{j}.ram'))
    return dict(status='PASS',run=run.name,seed=plan['seed'],banks=nb,samples=count,learning_mask=mask,configured_learning_neurons=nb*64,measured_bank_snapshots=nb*count,derived_response_bits=nb*count*64,bank_connection_bit_edits=[b.edits for b in banks],bank_changed_neuron_records=[len(b.records) for b in banks],exact_output_values=2*count,fixed_memory_unchanged=True,intermediate_activations='reconstructed, not measured',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1];results=[]
    for r in sorted((a.study/'raw').iterdir()):
        if (r/'RECEIPT.json').exists() and json.loads((r/'RECEIPT.json').read_text())['status']=='PASS':results.append(verify(root,r))
    z=dict(status='PASS',runs=len(results),results=results);(a.study/'evidence/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z))
