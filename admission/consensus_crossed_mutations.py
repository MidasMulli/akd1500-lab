"""Reject wrong source identity, invented recall and forged program ancestry."""
import copy,json
import numpy as np
from admission import consensus_archive as parent
from admission.consensus_crossed_archive import BASE,audit,get_run,validate_ancestry,mod
from admission.run import save

def main():
    audit();m=json.loads(BASE.with_suffix('.json').read_text());attempts=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        r,get,fg=get_run(z,m,'native_s124101_native_crossed');receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']));a=mod.simulate([fg(f'bank{k}.ram') for k in range(3)],'crossed','native');n=len(a['events'])
        ground=next(e['index'] for e in a['events'] if e['phase']=='bad_source_2' and e['role']=='ground')
        exam=next(e['index'] for e in a['events'] if e['phase']=='protected_exam_2' and e['key']==0)
        for name,key,shape,index,bit in [('wrong_source_partition_after_swap','descriptors',(n,16),(ground,8),2<<16),('invented_recall_after_a_is_corrupted','scores',(n,128),(exam,88),1),('uncaptured_source_weight_edit','states0',(n,-1),(ground,2+4),1)]:
            v=get(key).copy().reshape(shape);v[index]^=bit
            try:mod.check(receipt,lambda k:v.ravel() if k==key else get(k),fg)
            except AssertionError as e:attempts.append(dict(name=name,status='REJECTED',reason=str(e)))
            else:raise AssertionError('corruption accepted: '+name)
    forged=copy.deepcopy(m);f=next(iter(forged['fixtures'].values()));f['metadata']['inherited_files']['model.bin']='0'*64
    try:validate_ancestry(forged,json.loads(parent.BASE.with_suffix('.json').read_text()))
    except AssertionError as e:attempts.append(dict(name='forged_reviewed_program_ancestry',status='REJECTED',reason=str(e)))
    else:raise AssertionError('forged ancestry accepted')
    out=dict(status='SEMANTIC_MUTATIONS_PASS',rejected=len(attempts),attempts=attempts,scope='In-memory corruptions bypass digest checks; saved sources unchanged; no device access.')
    save(BASE.with_suffix('.mutations.json'),out);print(json.dumps(out))
if __name__=='__main__':main()
