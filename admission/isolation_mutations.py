"""Rehashed semantic-corruption checks for the two-bank isolation archive."""
import json
import numpy as np
from admission.isolation_archive import BASE, audit, get_run
from admission.isolation import simulate
from admission import isolation_run as mod
from admission.run import save

def main():
    audit(); m=json.loads(BASE.with_suffix('.json').read_text()); attempts=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        name='native_s114101_native_repeat_w117000'; r,get,fg=get_run(z,m,name)
        receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']))
        a=simulate([fg(f'bank{k}.ram') for k in range(2)],'repeat','native'); n=len(a['events'])
        event=next(e['index'] for e in a['events'] if e['teacher'] and e['context']==1)
        negative=next(e['index'] for e in a['events'] if e['erase'])
        def check(override): return mod.check(receipt,lambda k:override[k] if k in override else get(k),fg)
        check({})
        def reject(label,change):
            o={};change(o)
            try: check(o)
            except AssertionError as e: attempts.append(dict(name=label,status='REJECTED',reason=str(e)));return
            raise AssertionError('Accepted corruption: '+label)
        def mutate(o,key,shape,index,value=None):
            x=get(key).copy().reshape(shape)
            if value is None:x[index]^=1
            else:x[index]=value
            o[key]=x.ravel()
        reject('foreign_context_RAM_changed',lambda o:mutate(o,'states0',(n,-1),(event,6)))
        reject('host_disables_foreign_bank',lambda o:mutate(o,'learning_modes',(n,2),(event,0),0))
        reject('host_reconstructs_prediction',lambda o:mutate(o,'inputs',(n,128),(event,84)))
        reject('context_output_swapped',lambda o:mutate(o,'scores',(n,128),(event-1,24)))
        reject('fixed_route_changed',lambda o:mutate(o,'fixed3.final',(-1,),-1))
        reject('native_packet_disagrees',lambda o:mutate(o,'packet0',(-1,),29))
        def lineage(o):
            x=get('input_addresses').copy(); x[negative]+=256; o['input_addresses']=x
            d=get('descriptors').copy().reshape(n,16);d[negative,1]=x[negative];o['descriptors']=d.ravel()
        reject('coupled_descriptor_and_source_pointer',lineage)
        reject('negative_descriptor_class_changed',lambda o:mutate(o,'descriptors',(n,16),(negative,8),int(get('descriptors').reshape(n,16)[negative,8])^(1<<16)))
    result=dict(status='SEMANTIC_MUTATIONS_PASS',unmodified='PASS',rejected=len(attempts),attempts=attempts,
        scope='In-memory mutations bypass hashes. Saved measurements unchanged; no device access.')
    save(BASE.with_suffix('.mutations.json'),result); print(json.dumps(result))
if __name__=='__main__':main()
