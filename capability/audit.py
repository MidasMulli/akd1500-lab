"""Independent raw-array checks for matched primitives and quiet-host learning."""
import json
import numpy as np
from capability.design import S,weights,expected,DENSITIES
from resident.or_design import Network


def fixed(rec,z):
    x=z['inputs'];y=expected(x,weights()[:,:rec['units']]);assert np.array_equal(z['scores'],y),'fixed native output'
    idle=z['idle_counter'];assert len(idle)==6 and len(set(map(int,idle[:4])))==len(set(map(int,idle[4:])))==1,'counter idle controls'
    delta=(z['counter_bounds'][:,1].astype(np.uint64)-z['counter_bounds'][:,0].astype(np.uint64))&0xffffffff
    assert np.array_equal(delta,z['cycles']) and np.all(z['cycles']>0),'counter arithmetic'
    assert np.all(z['config_bounds'][:,0]==z['config_bounds'][:,1]),'configuration counter'
    for j in range(len(rec['fixture_info']['packets'])):
        assert np.array_equal(z[f'packet{j}.initial'],z[f'packet{j}.expected']),'initial fixed RAM'
        assert np.array_equal(z[f'packet{j}.final'],z[f'packet{j}.expected']),'final fixed RAM'
    for mode in ['serial_suffix','burst_suffix']:assert np.array_equal(z[mode],np.tile(y[-8:][None,:,:],(3,1,1))),'queued output'
    rows=[]
    for density in DENSITIES:
        ii=np.flatnonzero(x.sum(1)==density);assert len(ii)==(8 if rec['smoke'] else 32)
        rows.append(dict(density=density,samples=len(ii),median_cycles=float(np.median(z['cycles'][ii])),median_wall_ns=float(np.median(z['latency_ns'][ii])),median_caller_ns=float(np.median(z['caller_ns'][ii]))))
    return dict(status='INDEPENDENT_FIXED_PASS',cell=rec['cell'],samples=len(x),forwards=len(x)+48,groups=rows)


def quiet(rec,z):
    chunk=rec['chunk'];boundaries=list(range(512))+list(range(511+chunk,640,chunk))+list(range(640,768))
    assert np.array_equal(z['boundary_indices'],boundaries),'all quiet boundary captures'
    net=Network([z[f'bank{k}.initial'].tolist() for k in range(3)],512);at=0;old=None;metrics={};last=[z[f'bank{k}.initial'] for k in range(3)];changes=[]
    for i,row in enumerate(rec['rows']):
        y,_=net.forward(z['inputs'][i],int(z['labels'][i]),int(z['masks'][i]));assert np.array_equal(y,z['scores'][i]),('quiet output',i)
        if row['phase'] in ['old','changed_before','changed_after','retention']:
            m=metrics.setdefault(row['phase'],dict(correct=0,count=0));m['count']+=1;m['correct']+=int(np.argmax(y)==row['destination'])
        if i==511:old=[b.ram[:] for b in net.banks]
        if i in boundaries:
            for k,b in enumerate(net.banks):
                got=z[f'bank{k}.snapshots'][at];assert np.array_equal(got,b.ram),('quiet complete state',i,k)
                if 512<=i<640 and k>0:assert np.array_equal(got,old[k]),'protected downstream RAM'
                if 512<=i<640 and k==0:
                    aa=last[k][2:].reshape(512,20)[:,4:];bb=got[2:].reshape(512,20)[:,4:];changes.append(dict(index=i,records=np.flatnonzero((aa!=bb).any(1)).tolist()))
                last[k]=got
            at+=1
    assert at==len(boundaries)
    for k,b in enumerate(net.banks):
        assert np.array_equal(z[f'bank{k}.final'],b.ram),'quiet final learning state'
        assert np.array_equal(z[f'fixed{k}.initial'],z[f'fixed{k}.expected']) and np.array_equal(z[f'fixed{k}.final'],z[f'fixed{k}.expected']),'quiet fixed memory'
    t=z['released_intervals_ns'];a=z['host_activity'];c=z['counter_jobs'];assert len(t)==128//chunk and t.shape==a.shape==(128//chunk,5) and c.shape==(128//chunk,4)
    assert np.array_equal(t[:,0],np.arange(512,640,chunk)) and np.all(t[:,1]==chunk) and np.all(t[:,2]==rec['sleep_ms']),'released interval labels'
    assert np.all(c[:,2]==c[:,3]) and np.all(c[:,2]>0),'completed target job'
    assert np.all(a[:,1]==1),'release-only host writes'
    if rec['sleep_ms']:
        assert np.all(a[:,0]==0) and np.all(a[:,2]>0) and np.all(a[:,4]==0),'blocked host thread without driver polling'
        assert np.all(t[:,3]>=rec['sleep_ms']*1000000),'declared sleep interval'
    assert all(metrics[p]['correct']==64 for p in ['old','changed_after','retention']),'quiet application retention'
    return dict(status='INDEPENDENT_QUIET_PASS',forwards=768,complete_bank_snapshots=3*len(boundaries),chunk=chunk,sleep_ms=rec['sleep_ms'],metrics=metrics,
        median_released_wall_ns=float(np.median(t[:,3])),median_released_thread_cpu_ns=float(np.median(t[:,4])),total_released_thread_cpu_ns=int(t[:,4].sum()),
        host_driver_reads_during_sleep=int(a[:,0].sum()) if rec['sleep_ms'] else None,first_wake_completed=bool(np.all(c[:,2]==c[:,3])),repair_boundary_connection_changes=changes)


def cpu(rec,z):
    x=z['inputs'];y=expected(x,weights()[:,:rec['units']]);assert z['scores'].shape==(3,160,2)
    assert np.array_equal(z['scores'],np.tile(y[None,:,:],(3,1,1))),'CPU exact primitive outputs'
    assert z['wall_ns'].shape==z['cpu_ns'].shape==(3,3,5) and np.all(z['wall_ns']>0) and np.all(z['cpu_ns']>0)
    assert rec['result']['neon'] and rec['result']['timed_calls_per_density_sample']==8192
    rows=[]
    for method,name in enumerate(['packed_neon','inverted_index','fixed_adaptive']):
        for d,density in enumerate(DENSITIES):
            rows.append(dict(method=name,density=density,median_wall_ns=float(np.median(z['wall_ns'][:,method,d])/8192),median_thread_cpu_ns=float(np.median(z['cpu_ns'][:,method,d])/8192),temporal_wall_ns_per_input=(z['wall_ns'][:,method,d]/8192).tolist()))
    return dict(status='INDEPENDENT_CPU_PASS',units=rec['units'],groups=rows,timed_calls=3*3*5*8192)
