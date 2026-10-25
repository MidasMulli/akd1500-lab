"""Reject semantic corruptions of native transfer, beyond matching hashes."""
import json
import numpy as np
from admission.transfer_archive import BASE,audit,get_run
from admission.transfer import simulate
from admission import transfer_run as mod
from admission.run import save
def main():
    audit();m=json.loads(BASE.with_suffix('.json').read_text());attempts=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        r,get,fg=get_run(z,m,'raw02__native_s118101_native_extended');receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']))
        a=simulate([fg(f'bank{k}.ram') for k in range(3)],'extended','native');n=len(a['events'])
        transfer=next(e['index'] for e in a['events'] if e['role']=='transfer');exam=next(e['index'] for e in a['events'] if e['role']=='exam');unknown=next(e['index'] for e in a['events'] if e['role']=='transfer' and e['phase']=='unknown_transfer')
        def check(o):return mod.check(receipt,lambda k:o[k] if k in o else get(k),fg)
        def mutate(o,key,shape,index,value=None):
            x=get(key).copy().reshape(shape)
            if value is None:x[index]^=1
            else:x[index]=value
            o[key]=x.ravel()
        def reject(name,fn):
            o={};fn(o)
            try:check(o)
            except AssertionError as e:attempts.append(dict(name=name,status='REJECTED',reason=str(e)));return
            raise AssertionError('corruption accepted: '+name)
        check({})
        reject('nonconstant_host_transfer_class',lambda o:mutate(o,'descriptors',(n,16),(transfer,8),int(get('descriptors').reshape(n,16)[transfer,8])^(3<<16)))
        reject('host_injected_student_recall',lambda o:mutate(o,'inputs',(n,128),(transfer,88)))
        reject('teacher_answer_present_during_silent_exam',lambda o:mutate(o,'scores',(n,128),(exam,84),1))
        reject('unknown_source_changes_student_RAM',lambda o:mutate(o,'states1',(n,-1),(unknown,6)))
        reject('host_selects_only_one_student',lambda o:mutate(o,'learning_modes',(n,3),(transfer,2),0))
        reject('fixed_valence_bridge_changed',lambda o:mutate(o,'fixed3.final',(-1,),-1))
        reject('native_packet_disagrees',lambda o:mutate(o,'packet0',(-1,),29))
        def lineage(o):
            x=get('input_addresses').copy();x[transfer]+=256;o['input_addresses']=x
            d=get('descriptors').copy().reshape(n,16);d[transfer,1]=x[transfer];o['descriptors']=d.ravel()
        reject('coupled_descriptor_source_pointer',lineage)
    result=dict(status='SEMANTIC_MUTATIONS_PASS',unmodified='PASS',rejected=len(attempts),attempts=attempts,scope='In-memory corruptions bypass digest checks. No device access or saved-measurement edits.')
    save(BASE.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
