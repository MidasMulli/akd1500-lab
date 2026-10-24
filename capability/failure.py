"""Recount the measured prefix of the preserved129-slot failure; no rerun."""
import json
import numpy as np
from capability.design import P,S
from capability.run import digest,save
from resident.or_design import Network,schedule


def recount(rec,z):
    assert rec['status']=='FAILED_PRESERVED' and rec['exit_code']==-11
    assert rec['failure_stage']=='requested129_capacity_guard_before_repair'
    assert z['scores'].shape==(512,4) and z['inputs'].shape==(512,512)
    net=Network([z[f'bank{k}.initial'].tolist() for k in range(3)],512);x,rows=schedule('repair');assert np.array_equal(x[:512],z['inputs'])
    for i in range(512):
        row=rows[i];y,_=net.forward(x[i],row['label'],row['mask']);assert np.array_equal(y,z['scores'][i]),('failed prefix output',i)
        for k,b in enumerate(net.banks):assert np.array_equal(b.ram,z[f'bank{k}.snapshots'][i]),('failed prefix full RAM',i,k)
    for k in range(3):assert np.array_equal(z[f'fixed{k}.initial'],z[f'fixed{k}.expected']),'failed prefix initial fixed RAM'
    return dict(status='VERIFIED_PREFIX_CAPACITY_FAILURE_PRESERVED',completed_native_forwards=512,requested_native_forwards=768,complete_bank_snapshots=1536,
        repair_forwards=0,exit_code=-11,device_handle_released=rec['device_released'],original_unprogram_completion='unconfirmed',
        boundary='No successful128-event learning result. Runtime clamp measured separately; caller program-buffer lifetime hazard found, SIGSEGV cause not uniquely isolated.')


def raw(d):
    rec=json.loads((d/'RECEIPT.json').read_text());rec['failure_stage']='requested129_capacity_guard_before_repair'
    for p,h in rec['hashes'].items():assert digest(d/p)==h,p
    assert 'quiet requested queue capacity' in (d/'native.stderr').read_text()
    z=dict(inputs=np.fromfile(d/'stream.inputs',np.uint8).reshape(768,512)[:512],scores=np.fromfile(d/'native.scores','<i4').reshape(-1,4))
    for k in range(3):
        z[f'bank{k}.initial']=np.fromfile(d/f'native.bank{k}.initial','<u4');z[f'bank{k}.snapshots']=np.fromfile(d/f'native.bank{k}.snapshots','<u4').reshape(-1,10242)
        z[f'fixed{k}.initial']=np.fromfile(d/f'native.fixed{k}.initial','<u4');z[f'fixed{k}.expected']=np.fromfile(P/(rec['fixture']+f'.fixed{k}.ram'),'<u4')
    r=recount(rec,z);save(d/'AUDIT_FAILED_PREFIX.json',r);return rec,z,r


if __name__=='__main__':print(json.dumps(raw(S/'raw/quiet_b128_s250_r0')[2]))
