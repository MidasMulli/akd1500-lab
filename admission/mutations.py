"""Rehashed, valid-shaped corruptions must fail semantic evidence checks."""
import copy,json
import numpy as np
from admission.archive import BASE,digest,read_run,save
from admission.learned_run import check

def main():
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        r,get,fg=read_run(z,'learned','native_s114101_native_heldout');base={k:get(k).copy() for k in r['arrays']};m=json.loads(BASE.with_suffix('.json').read_text());f=m['families']['learned']['fixtures']['s114101_native'];fixtures={k:fg(k).copy() for k in f['arrays']}
    check(r['receipt'],base.__getitem__,fixtures.__getitem__)
    def permission(a):a['inputs'].reshape(-1,128)[3,80]^=1
    def manager(a):a['states0'].reshape(-1,386)[0,6]^=1
    def protected(a):a['states2'].reshape(-1,1154)[320,6]^=1
    def label(a):a['descriptors'].reshape(-1,16)[0,8]=(a['descriptors'].reshape(-1,16)[0,8]&np.uint32(0xfc00ffff))|np.uint32(2<<16)
    def pointer(a):a['descriptors'].reshape(-1,16)[3,1]+=4;a['input_addresses'][3]+=4
    def raw(a):a['inputs'].reshape(-1,128)[0,32]^=1
    def mode(a):a['learning_modes'].reshape(-1,3)[0,0]^=1
    def returned(a):a['scores'].reshape(-1,128)[2,80]^=1;a['packet2'].view('u1')[32+80]^=1
    def fixed(a):a['fixed1.final'][9]^=1
    rows=[]
    for name,fn in [('raw_observation',raw),('manager_synapses',manager),('native_permission_input',permission),('returned_permission_and_packet',returned),('phase_clocks',mode),('future_label_leak',label),('causal_teacher_source',pointer),('protected_recall_ram',protected),('fixed_policy_ram',fixed)]:
        a={k:v.copy() for k,v in base.items()};fn(a)
        # Update all content hashes: rejection must be semantic, not a stale hash.
        changed=copy.deepcopy(r)
        for k,v in a.items():
            changed['arrays'][k]['sha256']=digest(v)
            changed['receipt']['hashes']['result.'+k]=digest(v)
        assert all(digest(a[k])==v['sha256'] for k,v in changed['arrays'].items())
        assert all(digest(v)==changed['receipt']['hashes']['result.'+k] for k,v in a.items())
        try:check(changed['receipt'],a.__getitem__,fixtures.__getitem__)
        except AssertionError as e:rows.append(dict(name=name,status='REJECTED_AFTER_REHASH',reason=str(e)))
        else:raise AssertionError('accepted semantic corruption: '+name)
    result=dict(status='PASS',mutations=rows);save(BASE.with_name(BASE.name+'-mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
