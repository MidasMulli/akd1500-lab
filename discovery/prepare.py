"""Virtual/compiler-only sequential context allocator with fixed history helpers."""
from pathlib import Path
import argparse,json,sys,traceback,hashlib
import akida,numpy as np
from discovery.model import *
from tools.prepare import parts
from metal.prepare import mapping,add_fc
from coprocessor.resident_prepare import packets
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select

def main(out,seed):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,physical_device_opened=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        enc,relay=matrices();w=weights(seed)
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
        feat=add_fc(m,'fixed_three_bit_conjunctions',enc,True);feat.set_variable('threshold',np.full(FEATURES,2,np.int32))
        learner=akida.FullyConnected(name='unlabelled_context_bank_and_fixed_history_helpers',units=U,weights_bits=1,activation=True,act_bits=1);m.add(learner)
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=1,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=0))
        learner.set_variable('weights',w.reshape(1,1,FEATURES,U));learner.set_variable('threshold_learning',np.r_[np.zeros(LEARNED),np.full(8,32)].astype('i4'));learner.set_variable('threshold',np.r_[np.full(LEARNED,27),np.full(8,7)].astype('i4'));learner.set_variable('act_step',np.ones(U,np.float32))
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single)
        assert np.array_equal(np.asarray(ram)[[2+j*12+3 for j in range(U)]],np.r_[np.full(LEARNED,16),np.full(8,(32<<16)|16)])
        m.map(None);joined=add_fc(m,'fixed_history_and_context_output',relay,True);joined.set_variable('threshold',np.zeros(N,np.int32))
        x=np.zeros((16,N),'u1')
        for h in range(16):
            for k in range(4):x[h,k*16+((h>>k)&1)*8:k*16+((h>>k)&1)*8+8]=1
        features=(x.astype('i4')@enc.astype('i4')>2).astype('u1');assert np.all(features.sum(1)==32)
        v=np.zeros((16,N),'u1');v[:,:48]=x[:,16:64];v[:,64:96]=(features.astype('i4')@w[:,:LEARNED].astype('i4')>27)
        got=m.forward(x.reshape(16,1,1,N)).reshape(16,N);assert np.array_equal(got,v),'native helper shift semantics'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,adapter=assemble(single,bytes(m.sequences[0].program),FEATURES,U);pm=packets(program,npid);assert len(pm)==2
        (out/'model.bin').write_bytes(program);np.asarray(ram,'<u4').tofile(out/'initial.ram')
        for i,(_,packet) in enumerate(pm):np.asarray(packet,'<u4').tofile(out/f'fixed{i}.ram')
        for k,vv in [('encoder',enc),('relay',relay),('weights',w)]:vv.tofile(out/k)
        cells={}
        for arm in ARMS:
            for cycles in STAGES:
                a=simulate(ram,cycles,arm);stem=out/f'{arm}_c{cycles}'
                np.asarray([[e['bit'],e['label'],e['sever']] for e in a['events']],'<i4').tofile(stem.with_suffix('.plan'))
                for k in ['inputs','features','outputs','potentials','states','cpu']:a[k].tofile(stem.with_suffix('.'+k))
                cells[f'{arm}_c{cycles}']=assess(a)
                if arm=='native' and cycles==12:assert cells[f'{arm}_c{cycles}']['unique_records']==16
        r.update(status='VIRTUAL_COMPILER_REFERENCE_PASS',mapping=mapping(m),learning_np=npid,adapter=adapter,ram_words=len(ram),fixed_packets=len(pm),cells=cells)
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);a=p.parse_args();main(a.out,a.seed)
