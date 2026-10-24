"""Rehashed semantic corruptions must fail the portable verifier."""
import copy,hashlib,json
import numpy as np
from discovery.archive import BASE,check,save

def main():
    m=json.loads(BASE.with_suffix('.json').read_text());name='native_s113101_unseen_c12';original=m['runs'][name];f=m['fixtures']['s113101']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as v:base={k:v[k] for k in v.files}
    rows=[]
    mutations=[('training_label_field','descriptors',99*16+8,1<<16),('carried_history','inputs',330*128,1),('connection_state','states',99*482+6,1),('query_learning_mode','learning_modes',330,1),('returned_activity','scores',330*128+64,1),('fixed_frontend','fixed0.final',10,1),('previous_output_pointer','input_addresses',330,128)]
    for label,key,at,bit in mutations:
        run=copy.deepcopy(original);z=base.copy();k=run['arrays'][key]['key'];z[k]=z[k].copy();z[k][at]^=bit
        run['arrays'][key]['sha256']=hashlib.sha256(z[k].tobytes()).hexdigest();run['receipt']['hashes']['result.'+key]=run['arrays'][key]['sha256']
        if key=='input_addresses':
            dk=run['arrays']['descriptors']['key'];z[dk]=z[dk].copy();z[dk].reshape(-1,16)[330,1]=z[k][330];run['arrays']['descriptors']['sha256']=hashlib.sha256(z[dk].tobytes()).hexdigest();run['receipt']['hashes']['result.descriptors']=run['arrays']['descriptors']['sha256']
        try:check(run,z,f)
        except AssertionError as e:rows.append(dict(mutation=label,status='REJECTED_AFTER_REHASH',reason=str(e)))
        else:raise AssertionError(('semantic corruption accepted',label))
    save(BASE.with_name(BASE.name+'-mutations').with_suffix('.json'),dict(status='SEMANTIC_MUTATIONS_REJECTED',count=len(rows),mutations=rows));print(json.dumps(rows))
if __name__=='__main__':main()
