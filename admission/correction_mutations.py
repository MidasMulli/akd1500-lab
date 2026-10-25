"""Adversarial semantic audit checks, beyond hashes, without hardware."""
from pathlib import Path
import json,tempfile
import numpy as np
from admission.correction_archive import BASE,audit
from admission import recycle_run as mod
from admission.recycle import simulate,SEEDS
from admission.run import save

def main():
    audit();m=json.loads(BASE.with_suffix('.json').read_text());fam=m['families']['correction'];r=fam['runs'][f'native_s{SEEDS[0]}_native_repeat'];f=fam['fixtures'][f's{SEEDS[0]}_native'];results=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z,tempfile.TemporaryDirectory() as td:
        arrays={k:z[v['key']].copy() for k,v in r['arrays'].items()};fixtures={k:z[v['key']].copy() for k,v in f['arrays'].items()}
        a=simulate(fixtures['bank.ram'],'repeat','native');rows=a['events'];count=len(rows)
        refused=next(e['index'] for e in rows if e['teacher'] and not a['features'][e['index']].any())
        negative=next(e['index'] for e in rows if e['erase'])
        receipt=dict(r['receipt'],directory=td);Path(td,'stdout').write_text(r['stdout'])
        def check(overrides):return mod.check(receipt,lambda k:overrides.get(k,arrays[k]),lambda k:fixtures[k])
        check({})
        def change(name,fn):
            overrides={};fn(overrides)
            try:check(overrides)
            except AssertionError as e:results.append(dict(name=name,status='REJECTED',reason=str(e)));return
            raise AssertionError('corruption accepted: '+name)
        def input_edit(d):
            x=arrays['inputs'].copy().reshape(count,128);x[refused,84]^=1;d['inputs']=x.ravel()
        def ram_edit(d):
            x=arrays['states'].copy().reshape(count,-1);x[refused,6]^=1;d['states']=x.ravel()
        def fixed_edit(d):
            x=arrays['fixed0.final'].copy();x[-1]^=1;d['fixed0.final']=x
        def class_edit(d):
            x=arrays['descriptors'].copy().reshape(count,16);x[negative,8]=(int(x[negative,8])&~(1023<<16))|((rows[negative]['actual']+1)<<16);d['descriptors']=x.ravel()
        def lineage_edit(d):
            x=arrays['input_addresses'].copy();x[negative]+=256;q=arrays['descriptors'].copy().reshape(count,16);q[negative,1]=x[negative];d['input_addresses']=x;d['descriptors']=q.ravel()
        def packet_edit(d):
            x=arrays['packet0'].copy();x[29]^=1;d['packet0']=x
        for name,fn in [('host_reconstructed_prediction',input_edit),('refused_teacher_ram_change',ram_edit),('fixed_comparator_change',fixed_edit),('negative_class_metadata_swapped_to_actual',class_edit),('coupled_descriptor_and_source_pointer',lineage_edit),('packet_response_disagrees_with_capture',packet_edit)]:change(name,fn)
    result=dict(status='SEMANTIC_MUTATIONS_PASS',unmodified='PASS',rejected=len(results),attempts=results,
        scope='In-memory numeric mutations bypass digest checks to test semantic rejection. No device access or modified saved measurements.')
    save(BASE.with_suffix('.mutations.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
