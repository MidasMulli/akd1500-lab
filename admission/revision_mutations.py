"""Semantic mutation checks for corrected native teacher/student evidence."""
import json
import numpy as np
from admission.revision_archive import BASE,audit,get_run
from admission import revision_run as mod
from admission.run import save
def main():
    audit();m=json.loads(BASE.with_suffix('.json').read_text());attempts=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        r,get,fg=get_run(z,m,'native_s122101_native_repeat');receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']));a=mod.simulate([fg(f'bank{k}.ram') for k in range(3)],'repeat','native');n=len(a['events'])
        transfer=next(e['index'] for e in a['events'] if e['role']=='transfer' and e['block']);negative=next(e['index'] for e in a['events'] if e['role']=='ground_negative');exam=next(e['index'] for e in a['events'] if e['phase']=='exam_1')
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
        reject('source_complement_replaced_by_actual_class',lambda o:mutate(o,'descriptors',(n,16),(negative,8),int(get('descriptors').reshape(n,16)[negative,8])^(3<<16)))
        reject('nonconstant_student_transfer_label',lambda o:mutate(o,'descriptors',(n,16),(transfer,8),int(get('descriptors').reshape(n,16)[transfer,8])^(3<<16)))
        reject('host_injected_student_prediction',lambda o:mutate(o,'inputs',(n,128),(transfer,88)))
        reject('teacher_answer_present_during_exam',lambda o:mutate(o,'scores',(n,128),(exam,84),1))
        reject('unrelated_original_record_changed',lambda o:mutate(o,'states1',(n,-1),(exam,2+5*20+4)))
        reject('host_selects_only_one_student',lambda o:mutate(o,'learning_modes',(n,3),(transfer,2),0))
        reject('fixed_valence_bridge_changed',lambda o:mutate(o,'fixed3.final',(-1,),-1))
        def lineage(o):
            x=get('input_addresses').copy();x[transfer]+=256;o['input_addresses']=x
            d=get('descriptors').copy().reshape(n,16);d[transfer,1]=x[transfer];o['descriptors']=d.ravel()
        reject('coupled_descriptor_source_pointer',lineage)
    result=dict(status='SEMANTIC_MUTATIONS_PASS',unmodified='PASS',rejected=len(attempts),attempts=attempts,scope='In-memory mutations bypass digest checks. Saved raw sources unchanged; no device access.')
    save(BASE.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
