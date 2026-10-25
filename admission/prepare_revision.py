"""Virtual compiler gate; never opens or discovers a physical device."""
from pathlib import Path
import argparse,json,struct,traceback
import akida,numpy as np
from admission.revision import *
from admission.prepare_recycle import fixed
from admission.prepare import mapping,parts
from tools.prepare_network import Select as ExistingSelect
class Select(ExistingSelect):
    def select_nps(self,source_nps,num_nps,type):
        if type==akida.NP.Type.FNP2:
            found=[q for q in source_nps if [q.col,q.row,q.id]==[1,2,0]]
            assert len(found)>=num_nps,(str(type),num_nps,len(found))
            return found[:num_nps]
        return super().select_nps(source_nps,num_nps,type)
from tools.program_codec import read,build,words

def bind(prefixes):
    h,data=read(prefixes[-1]);data=bytearray(data);banks=[]
    for k,prefix in enumerate(prefixes):
        hh,dd=read(prefix);lr=hh['learning'];ram=words(dd,lr['ram'][0]);inf=words(dd,lr['inference']);lea=words(dd,lr['learning']);stride=4+D//32
        if k==2:ni=words(data,h['learning']['inference']);nl=words(data,h['learning']['learning'])
        else:
            candidates=[(j,words(data,sp)) for j,rec in enumerate(h['passes'][0]) for sp in rec if len(words(data,sp))==52+len(ram) and words(data,sp)[0]==inf[0] and words(data,sp)[52:54]==ram[:2]]
            assert len(candidates)==1,(k,len(candidates))
            ri,packet=candidates[0];ni=packet[:52];nl=ni[:]
            assert all(j in [46,47,49] for j,(a,b) in enumerate(zip(inf,ni)) if a!=b)
            assert all(j>=2 and (j-2)%stride in [1,2,3] for j,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b)
            for j in [38,39,40]:assert ni[j]==inf[j];nl[j]=lea[j]
            sp=[len(data),52+len(ram)];data.extend(np.asarray(ni+ram,'<u4').tobytes());h['passes'][0][ri]=[sp]
        banks.append(dict(np=lr['np'],ram=ram,infer=ni,learn=nl))
    struct.pack_into('<I',data,0,len(data)-4);image=build(h,bytes(data));assert read(image)==(h,bytes(data));pm=[]
    for rec in h['passes'][0]:
        for span in rec:
            a=words(data,span);at=0
            while at<len(a):
                header,size=a[at:at+2];length=(size>>16)&0x3fff
                if size>>31:length*=16
                packet=a[at:at+2+length];assert len(packet)==2+length;at+=len(packet);mask,uid=(header>>8)&15,header&15
                if mask and uid in [0,2,4,6]:
                    assert not mask&(mask-1);npid=[header>>24,(header>>16)&255,mask.bit_length()-1]
                    if npid not in [b['np'] for b in banks]:pm.append(dict(np=npid,packet=packet))
            assert at==len(a)
    assert len(pm)==len({tuple(p['np']) for p in pm})==4
    return image,banks,pm

def main(out,seed,arm):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,arm=arm,physical_device_opened=False)
    try:
        mats=matrices(arm);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));prefixes=[]
        fixed(m,'source_key_role_and_student_recall_encoder',*mats[0],2)
        for k,(u,classes,plastic,helpers) in enumerate(DIMENSIONS):
            if k:
                for j in ([1,2] if k==1 else [3]):fixed(m,['source_key_role_and_student_recall_encoder','native_teacher_error_predicate','student_zero_valence_code','native_opposite_valence_bridge'][j],*mats[j],2 if j in [1,3] else 1)
            m.add(akida.FullyConnected(name=['learned_source_teacher','student_outcome_zero','student_outcome_one_and_readout'][k],units=u,weights_bits=1,activation=True,act_bits=1))
            m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=classes,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
            w=weights(seed,k);lt,t=thresholds(k);l=m.layers[-1]
            l.set_variable('weights',w.reshape(1,1,D,u));l.set_variable('threshold_learning',lt);l.set_variable('threshold',t);l.set_variable('act_step',np.ones(u,np.float32))
            r['compiler_stage']=k
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
            prefix=bytes(m.sequences[0].program);prefixes.append(prefix);ram,npid=parts(prefix)
            assert all(a==b for j,(a,b) in enumerate(zip(ram,packed(w,k))) if j>=2 and (j-2)%(4+D//32)>=3)
            m.map(None)
        rams=[parts(p)[0] for p in prefixes];test=simulate(rams,'gate',arm)['inputs'];v=test.astype('i4');v=(v@mats[0][0].astype('i4')>mats[0][1]).astype('i4')
        for k in range(3):
            v=(v@weights(seed,k).astype('i4')>thresholds(k)[1]).astype('i4')
            if k<2:
                for w,t in mats[1:3] if k==0 else mats[3:4]:v=(v@w.astype('i4')>t).astype('i4')
        assert np.array_equal(m.forward(test.reshape(-1,1,1,N)).reshape(-1,N),v),'complete virtual circuit'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        r['mapping']=mapping(m)
        program,banks,pm=bind(prefixes);(out/'model.bin').write_bytes(program)
        for k,b in enumerate(banks):
            for ext in ['ram','infer','learn']:np.asarray(b[ext],'<u4').tofile(out/f'bank{k}.{ext}')
            thresholds(k)[1].tofile(out/f'bank{k}.threshold')
        for k,b in enumerate(pm):np.asarray(b['packet'],'<u4').tofile(out/f'fixed{k}.ram')
        for k,(w,t) in enumerate(mats):w.tofile(out/f'fixed{k}.weights');t.tofile(out/f'fixed{k}.threshold')
        cells={}
        for stage in STAGES:
            rows=schedule(stage,arm);np.asarray([[e['key'],ROLES.index(e['role']),*e['labels'],e['wire'],e['block']] for e in rows],'<i4').tofile(out/f'{stage}.plan');cells[stage]=dict(events=len(rows))
        r.update(status='VIRTUAL_COMPILER_PASS',mapping=mapping(m),banks=[dict(np=b['np'],u=DIMENSIONS[k][0],classes=DIMENSIONS[k][1],words=len(b['ram'])) for k,b in enumerate(banks)],fixed_packets=[dict(np=b['np'],words=len(b['packet'])) for b in pm],cells=cells)
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int,choices=SEEDS);p.add_argument('arm',choices=ARMS);a=p.parse_args();main(a.out,a.seed,a.arm)
