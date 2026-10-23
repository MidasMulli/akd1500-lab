"""Independent raw MMIO/connection/task algebra audit. No compiler/builder import."""
from pathlib import Path
import csv,hashlib,json,struct,sys
P=Path(__file__).resolve().parent
def pack(w):return struct.pack('<'+'I'*len(w),*w)
def verify(d,allow_failed_host=False):
    rec=json.loads((d/'RECEIPT.json').read_text());assert rec['exit_code']==0
    assert not rec['failed_at_host'] or allow_failed_host
    assert hashlib.sha256((P/rec['program']).read_bytes()).hexdigest()==rec['program_sha256']
    assert hashlib.sha256((d/'commands.txt').read_bytes()).hexdigest()==rec['commands_sha256']
    binary_candidates=[P/'native_service',P/'evidence/COHORT_native_service']
    assert any(f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==rec['native_sha256'] for f in binary_candidates),'receipt binary missing or changed'
    audit=json.loads((P/rec['audit']).read_text());initial=[x['initial_words'] for x in audit['memories']]+[x['words'] for x in audit['fixed']]
    data=[json.loads(l) for l in (d/'stdout.jsonl').read_text().splitlines()];assert data[-1]==dict(kind='final',status='PASS',cleanup='unprogrammed')
    rows=list(csv.DictReader((d/'mmio.csv').open()));marks=[(i,r['data_hex']) for i,r in enumerate(rows) if r['operation']=='MARK']
    def mark(name):
        found=[i for i,n in marks if n==name];assert len(found)==1,(name,found);return found[0]
    def bound(op,lo,hi,address,value):
        assert any(r['operation']==op and int(r['address'])==address and r['data_hex']==value.hex() for r in rows[lo:hi]),('raw binding',op,address,len(value))
    # Reconstruct every config packet to reject any post-program neural-RAM write.
    mem={};registers=[];first=mark('snapshot_-1')
    for i,r in enumerate(rows):
        if r['operation']!='WRITE':continue
        address=int(r['address']);b=bytes.fromhex(r['data_hex'])
        if 0x20000100<=address<0x20100000:
            for k,value in enumerate(b):mem[address+k]=value
            if len(b)==32:
                ds=struct.unpack('<8I',b)
                if ds[0]>>28==1 and not any(ds[3:]):
                    packet=bytes(mem.get(ds[1]+k,0) for k in range((ds[2]&65535)*4))
                    if len(packet)>=8:
                        header=struct.unpack_from('<I',packet)[0]
                        if header&255==6:assert i<first,'host neural-RAM write after programming'
                        if header&255==8:registers.append((i,header,packet))
    ready=next(x for x in data if x['kind']=='ready');assert len(set(ready['slabs']))==2
    assert ready['allocated_bytes']==768 and ready['peak_bytes']>=ready['allocated_bytes']
    vector=[0]*32;vector[3]=1;carry=ready['initial_address'];mapping=[None]*8;history=[None]*4;pending=False
    mode=[1,1];last_snapshot=None;observed_since_snapshot=False;events=[];snapshots=[];queues=0;injection=None;starts=[];allowed_custom_writes=set()
    for v in data:
        if v['kind']=='mode':mode=[v['transition'],v['history']]
        elif v['kind']=='request_start':
            assert not pending;expected=[0]*64;expected[3]=1;assert v['bytes']==expected and v['address']==ready['initial_address']
            lo=mark('request_start_'+str(v['id']));hi=next((i for i,n in marks if i>lo),len(rows));bound('WRITE',lo,hi,v['address'],bytes(expected))
            allowed_custom_writes.update(i for i in range(lo,hi) if rows[i]['operation']=='WRITE' and int(rows[i]['address'])==v['address'] and rows[i]['data_hex']==bytes(expected).hex())
            vector=expected[:32];carry=v['address'];starts.append(v['id'])
        elif v['kind']=='probe_request':
            assert not pending and v['address']==carry+20 and v['before']==vector[20:24] and vector[20]==0
            value=vector[20:24];value[0]=1;assert v['bytes']==value
            lo=mark('probe_request_'+str(v['id']));hi=mark('chain_setup_'+str(v['id']))
            bound('WRITE',lo,hi,v['address'],bytes(value));vector[20]=1
            allowed_custom_writes.update(i for i in range(lo,hi) if rows[i]['operation']=='WRITE' and int(rows[i]['address'])==v['address'] and rows[i]['data_hex']==bytes(value).hex())
        elif v['kind']=='injection':
            assert pending and v['address']==carry+24 and v['before']==[0]*4 and vector[24:28]==[0]*4
            expected=[0]*4;expected[v['observed_state']]=1;assert v['bytes']==expected
            lo=mark('external_observation_'+str(v['id']));hi=mark('chain_setup_'+str(v['id']))
            bound('WRITE',lo,hi,v['address'],bytes(expected));vector[24:28]=expected;injection=v
            allowed_custom_writes.update(i for i in range(lo,hi) if rows[i]['operation']=='WRITE' and int(rows[i]['address'])==v['address'] and rows[i]['data_hex']==bytes(expected).hex())
        elif v['kind']=='snapshot':
            lo=mark('snapshot_'+str(v['id']));hi=next((i for i,n in marks if i>lo and n.startswith(('snapshot_','chain_setup_','request_start_','external_observation_','UNPROGRAMMED'))),len(rows))
            for w in v['words']:
                found=[i for i,r in enumerate(rows[lo:hi],lo) if r['operation']=='READ' and r['data_hex']==pack(w[2:]).hex()]
                assert found,'snapshot not bound to raw read'
                assert any(r['operation']=='WRITE' and r['data_hex']==pack(w[:2]).hex() for r in rows[lo:found[-1]]),'snapshot target header'
            if last_snapshot is None:assert v['words']==initial
            elif not observed_since_snapshot:assert v['words']==last_snapshot,'protected queries/request initialization changed full RAM'
            elif not mode[0]:assert v['words'][0]==last_snapshot[0],'frozen transition RAM changed'
            assert v['words'][2:]==initial[2:],'fixed neural weights changed'
            assert v['words'][0][162:]==initial[0][162:] and v['words'][1][82:]==initial[1][82:],'protected neuron records changed'
            for ctx,out in enumerate(mapping):
                w=v['words'][0];bits=[i for i in range(512) if w[6+20*ctx+i//32]>>(i%32)&1]
                target=list(range(28*ctx,28*ctx+28))+list(range(480,484) if out is None else range(224+4*out,228+4*out))
                assert bits==sorted(target),('transition RAM',ctx,out)
            for s,h in enumerate(history):
                w=v['words'][1];bits=[i for i in range(512) if w[6+20*s+i//32]>>(i%32)&1]
                target=list(range(24*s,24*s+24))+(list(range(480,488)) if h is None else list(range(96+4*h[0],100+4*h[0]))+list(range(112+4*h[1],116+4*h[1])))
                assert bits==sorted(target),('history RAM',s,h)
            last_snapshot=v['words'];observed_since_snapshot=False
            snapshots.append(dict(id=v['id'],mapping=mapping[:],history=[x[:] if x else None for x in history],sha256=hashlib.sha256(pack([z for w in v['words'] for z in w])).hexdigest()))
        elif v['kind']=='chain':
            queues+=1;phase=v['phase'];assert v['n']==(2 if phase=='Q' else 1)
            assert v['initial_address']==carry and v['allocated_bytes']==ready['allocated_bytes'] and v['peak_bytes']==ready['peak_bytes']
            assert [v['transition_learning'],v['history_learning']]==mode
            lo=mark('chain_setup_'+str(v['id']));release=next(i for i,n in marks if i>lo and n=='QUEUE_RELEASED')
            begin=mark('NO_HOST_IO_BEGIN_'+str(v['id']));end=mark('NO_HOST_IO_END_'+str(v['id']));poll=mark('first_completion_poll_'+str(v['id']));drain=mark('POST_COMPLETE_SDK_DRAIN_'+str(v['id']))
            assert lo<release<begin<end<poll<drain and all(r['operation']=='MARK' for r in rows[begin:end+1])
            assert not any(r['operation']=='WRITE' for r in rows[release+1:drain]),'host wrote inside released queue'
            assert v['wait_us'] in [50,20000]
            assert int(rows[end]['us'])-int(rows[begin]['us'])>=v['wait_us']
            statuses=[r for r in rows[poll:drain] if r['operation']=='READ' and int(r['address'])==0xfcc28010];assert len(statuses)==v['completion_polls'];status=statuses[-1]
            assert int.from_bytes(bytes.fromhex(status['data_hex']),'little')>>16==v['first_completion']==v['descriptors'][-1]['words'][0]>>16
            for k,m in enumerate(audit['memories']):
                expected=pack(m['learning' if mode[k] else 'inference']);header=m['learning'][0]
                assert next(packet for ri,h,packet in reversed(registers) if ri<lo and h==header)==expected,('mode',k)
            if phase=='Q':assert not pending
            else:assert pending and injection is not None and injection['id']==v['id']
            for step,slot in enumerate(v['slots']):
                ds=v['descriptors'][step];words=ds['words'];assert slot['address'] in ready['slabs']
                assert words[1]==(carry if step==0 else v['slots'][step-1]['address']+32) and words[7]==slot['address']
                assert (words[8]>>16)&1023==(2 if phase=='Q' else 1),'descriptor class'
                bound('WRITE',lo,release,ds['address'],pack(words));bound('READ',poll,drain,slot['address'],bytes(slot['after']))
                assert slot['after'][64:]==[165]*64 and slot['before'][64:]==[165]*64
                assert sum(vector[:16])==1;idx=vector[:16].index(1);y=[0]*32
                if idx<4:
                    assert phase=='Q' and step==0;s=idx
                    r1=[int(any(mapping[2*k+a]==0 for a in range(2))) for k in range(4)];direct=[int(mapping[2*s+a]==0) for a in range(2)]
                    y[4+s]=1;y[16]=vector[16];y[17]=1;y[24:28]=r1;y[28:30]=direct;y[30]=int(history[s] is not None and history[s][0]==1);y[31]=int(any(direct));y[20]=vector[20]
                    event=dict(phase='predecessor_query',state=s,predecessor_set=[k for k in range(4) if r1[k]],direct=direct)
                elif idx<8:
                    assert phase=='Q' and step==1;s=idx-4;direct=vector[28:30];targets={k for k in range(4) if vector[24+k]}
                    hop=[mapping[2*s+a] in targets if mapping[2*s+a] is not None else False for a in range(2)];last1=history[s] is not None and history[s][0]==1
                    if vector[20]:action=0 if last1 else 1;depth=0
                    elif any(direct):action=direct.index(1);depth=1
                    elif rec['policy']=='twohop' and any(hop):action=hop.index(True);depth=2
                    else:action=0 if last1 else 1;depth=0
                    y[8+2*s+action]=1;y[16]=vector[16];y[18]=1;y[20 if depth==1 else 21 if depth==2 else 22]=1;y[30]=int(last1)
                    event=dict(phase='choice',state=s,action=action,depth=depth,probe=bool(vector[20]),predicted_next_state=mapping[2*s+action],predecessor_set=sorted(targets))
                else:
                    assert phase=='O';ctx=idx-8;s,action=divmod(ctx,2);actual=vector[24:28].index(1);assert actual==v['observed_state']==injection['observed_state']
                    y[3 if actual==0 else actual]=1;y[16]=action;y[19]=1;y[23]=int(actual==0)
                    event=dict(phase='observation',state=s,action=action,outcome=actual,predicted_next_state=mapping[ctx],correct_prediction=mapping[ctx]==actual if mapping[ctx] is not None else None)
                    if mode[0]:mapping[ctx]=actual
                    if mode[1]:history[s]=[action,actual]
                    observed_since_snapshot=True
                assert slot['after'][32:64]==y,('native output mismatch',v['id'],step)
                event.update(chain_id=v['id'],step=step);events.append(event);vector=y
            carry=v['slots'][-1]['address']+32;pending=phase=='Q'
    ranges=[(ready['initial_address'],ready['initial_address']+64)]+[(a,a+128) for a in ready['slabs']]
    for i,r in enumerate(rows):
        if i<=first or r['operation']!='WRITE':continue
        a,b=int(r['address']),int(r['address'])+int(r['size'])
        if any(a<hi and b>lo for lo,hi in ranges):assert i in allowed_custom_writes,'unadmitted host write to native state/action buffer'
    return dict(status='NATIVE_PREFIX_VALID_APPLICATION_FAILED' if rec['failed_at_host'] else 'PASS',run=d.name,native_forwards=len(events),released_queues=queues,host_writes_in_released_queues=0,
        allocated_bytes=ready['allocated_bytes'],peak_bytes=ready['peak_bytes'],fixed_slabs=ready['slabs'],
        native_action_selections=sum(x['phase']=='choice' for x in events),observations=sum(x['phase']=='observation' for x in events),
        request_starts=starts,all_fixed_RAM_unchanged=True,protected_neurons_unchanged=116,events=events,snapshots=snapshots,
        raw_sha256=hashlib.sha256((d/'mmio.csv').read_bytes()).hexdigest())
if __name__=='__main__':
    for name in sys.argv[1:]:
        d=P/'runs'/name;obj=verify(d);(d/'VERIFICATION_NATIVE.json').write_text(json.dumps(obj,indent=2)+'\n')
        print(json.dumps({k:v for k,v in obj.items() if k not in ['events','snapshots']}))
