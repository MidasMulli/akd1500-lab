"""Prospectively selected SDK source-bitwidth candidate and held-out qualification."""
import datetime as dt
import fcntl
import json
import shutil
import subprocess
import numpy as np
from resident.contract_design import P,S,data,comparison,expected
from resident.contract import arrays,inspect
from resident.run import save,digest


def prepare():
    import akida
    from resident.prepare import Select,conv
    from tools.prepare_network import fc
    from metal.prepare import mapping
    from metal.ablation_prepare import memory_packets
    out=S/'fixtures-bits02';out.mkdir(exist_ok=False);results=[]
    for bits in [2,4]:
        for component in ['identity','compare']:
            kind=f'mixed_{component}_bits{bits}';m=akida.Model();m.add(akida.InputData(input_shape=(1,1,512),input_bits=1))
            m.add(akida.FullyConnected(name='source',units=64,weights_bits=1,activation=True,act_bits=bits));l=m.get_layer('source')
            w=np.zeros((512,64),np.int8);w[:64]=np.eye(64,dtype=np.int8);l.set_variable('weights',w.reshape(1,1,512,64));l.set_variable('threshold',np.zeros(64,np.int32));l.set_variable('act_step',np.ones(64,np.float32))
            w,t=comparison() if component=='compare' else (np.eye(64,dtype=np.int8),np.zeros(64,np.int32));width=w.shape[1];conv(m,'bridge',w,t)
            fc(m,'readout',np.eye(width,dtype=np.int8),None,activation=False,bits=1)
            for split in ['development','sealed']:
                features=data(split);x=np.zeros((128,512),np.uint8);x[:,:64]=features;y=expected(kind,features)
                m.map(None);actual=m.forward(x.reshape(-1,1,1,512)).reshape(-1,width);assert np.array_equal(actual,y),(kind,split,'virtual equivalence')
                (out/f'{kind}.{split}.inputs').write_bytes(x.tobytes());(out/f'{kind}.{split}.expected').write_bytes(y.astype('<i4').tobytes())
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
            assert len(m.sequences)==len(m.sequences[0].passes)==1
            mp=mapping(m);assert all(p['type']!='Type.FNP2' for l in mp for p in l['nps'])
            program=bytes(m.sequences[0].program);packets=memory_packets(program);f=out/kind
            f.with_suffix('.bin').write_bytes(program);layout=[f'512 {width} 128 {len(packets)}'];meta=[]
            for j,pkt in enumerate(packets):
                p=out/f'{kind}.packet{j}.ram';p.write_bytes(np.asarray(pkt['packet'],'<u4').tobytes());layout.append(' '.join(map(str,pkt['np']+[len(pkt['packet'])])));meta.append(dict(np=pkt['np'],words=len(pkt['packet']),sha256=digest(p)))
            f.with_suffix('.layout').write_text('\n'.join(layout)+'\n');r=dict(kind=kind,status='COMPILER_VIRTUAL_PASS',source_act_bits=bits,input_features=512,outputs=width,mapping=mp,packets=meta,program_sha256=digest(f.with_suffix('.bin')))
            save(f.with_suffix('.json'),r);results.append(r);save(out/'BUILD.json',dict(status='COMPILER_ONLY',physical_device_opened=False,results=results));print(json.dumps(dict(kind=kind,status=r['status'])),flush=True)


def freeze():
    files=[P/f for f in ['resident/contract_candidate.py','resident/contract.py','resident/contract_design.py','resident/prepare.py',
        'tools/prepare_network.py','tools/program_codec.py','metal/ablation_prepare.py','src/contract_probe.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','build/contract_probe']]
    files+=list((S/'fixtures-bits02').iterdir())+[S/'PROTOCOL.md',S/'AUTHORITY.json',S/'evidence/CANDIDATE_FREEZE.json']
    p=S/'evidence/FROZEN02.json';assert not p.exists();save(p,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(f.relative_to(P)):digest(f) for f in files}))


def run(component,split,smoke,review):
    assert component in ['identity','compare'] and split in ['development','sealed'] and len(review)>20
    assert not (split=='sealed' and smoke)
    state=json.loads((S/'STATE.json').read_text());auth=json.loads((S/'AUTHORITY.json').read_text());assert not state['research_admissions_closed'] and not state.get('halted')
    kind=f'mixed_{component}_bits2';name=f'{kind}_{split}_{"smoke" if smoke else "full"}'
    if component=='compare':assert json.loads((S/'raw/mixed_identity_bits2_sealed_full/CHECK.json').read_text())['semantics_exact']
    if not smoke:
        before=f'{kind}_development_{"smoke" if split=="development" else "full"}'
        assert json.loads((S/'raw'/before/'CHECK.json').read_text())['semantics_exact']
    assert name not in [r['name'] for r in state['admitted_runs']]
    fr=S/'evidence/FROZEN02.json'
    for p,h in json.loads(fr.read_text())['hashes'].items():assert digest(P/p)==h,('candidate source drift',p)
    now=dt.datetime.now(dt.timezone.utc);cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cutoff-now).total_seconds()>5
    count=8 if smoke else 128;prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert len(prior)<auth['maximum_physical_sessions'] and sum(r['samples'] for r in prior)+count<=auth['maximum_forwards']
    entry=dict(name=name,kind=kind,split=split,smoke=smoke,review=review,admitted_utc=now.isoformat());state['admitted_runs'].append(entry);save(S/'STATE.json',state)
    f=S/'fixtures-bits02'/kind;d=S/'raw'/name
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;d.mkdir(exist_ok=False)
        for p in f.parent.glob(kind+'.*'):
            if p.suffix in ['.bin','.layout'] or '.packet' in p.name:shutil.copy2(p,d/('fixture'+p.name[len(kind):]))
        for ext in ['inputs','expected']:shutil.copy2(f.parent/f'{kind}.{split}.{ext}',d/f'fixture.{ext}')
        rec=dict(**entry,samples=count,start_utc=now.isoformat(),fixture=json.loads(f.with_suffix('.json').read_text()),source_freeze_sha256=digest(fr));save(d/'OWNER.json',rec)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            proc=subprocess.Popen([str(P/'build/contract_probe'),str(d/'fixture'),str(d/'native'),str(count)],stdout=out,stderr=err)
            try:code=proc.wait(timeout=min(120,int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        rec.update(exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_released'],(d/'native.stderr').read_text();check=inspect(rec,arrays(d,rec));save(d/'CHECK.json',check)
            if not check['semantics_exact']:state.update(halted=True,failure=name)
        except Exception:
            state.update(halted=True,failure=name);save(S/'STATE.json',state);raise
        save(S/'STATE.json',state);print(json.dumps({k:v for k,v in check.items() if k not in ['native_outputs','nonzero_outputs_per_vector']}),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':run(sys.argv[2],sys.argv[3],sys.argv[4]=='smoke',sys.argv[5])
    else:raise ValueError(sys.argv[1])
