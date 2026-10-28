"""Portable raw-numeric learned-policy archive; requires no SDK or board."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
from admission import learned_policy_run as run
from admission.run import save, sha

P=Path(__file__).resolve().parents[1]; BASE=P/'results/20261025-learned-policy'


def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()


def build():
    arrays={}; fixtures={}; runs={}
    for folder in sorted(run.F.iterdir()):
        if not (folder/'BUILD.json').exists():continue
        ff={}
        for path in sorted([*folder.glob('*.ram'),*folder.glob('*.threshold')]):
            key=f'{folder.name}__{path.name.replace(".","_")}'
            arrays[key]=np.fromfile(path,'<u4');ff[path.name]=dict(key=key,sha256=sha(path))
        fixtures[folder.name]=dict(arrays=ff,build=json.loads((folder/'BUILD.json').read_text()))
    for folder in sorted(run.RAW.iterdir()):
        if not (folder/'AUDIT.json').exists():continue
        ff={};packets=[]
        for path in sorted(folder.glob('result.*')):
            field=path.name[7:]
            if field.startswith('packet'):
                packets.append((int(field[6:]),np.fromfile(path,'<u4')));continue
            key=f'{folder.name}__{field.replace(".","_")}'
            arrays[key]=np.fromfile(path,run.dtype(field));ff[field]=dict(key=key,sha256=sha(path))
        packet_meta=None
        if packets:
            packets.sort(key=lambda row:row[0]);assert [r[0] for r in packets]==list(range(len(packets)))
            key=f'{folder.name}__packets';arrays[key]=np.stack([r[1] for r in packets]);packet_meta=dict(key=key,sha256=digest(arrays[key]))
        runs[folder.name]=dict(receipt=json.loads((folder/'RECEIPT.json').read_text()),arrays=ff,packets=packet_meta,
                              freeze=json.loads((folder/'FROZEN.json').read_text()),stdout=(folder/'stdout').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,
        state=json.loads((run.S/'STATE.json').read_text()),authority=json.loads((run.S/'AUTHORITY.json').read_text()),
        scope='Externally supervised learned source/context reliability controls native admission; designed keys, fixed schedule/objective; no real-sensor, efficiency, autonomy or learned eviction claim'))
    return audit()


def check_one(z,meta,name,corrupt=None):
    item=meta['runs'][name];r=item['receipt'];f=meta['fixtures'][f"s{r['seed']}_{r['arm']}"]['arrays']
    def get(field):
        a=z[item['packets']['key']][int(field[6:])] if field.startswith('packet') else z[item['arrays'][field]['key']]
        if field==corrupt:a=a.copy();a.flat[0]^=1
        return a
    result=run.check(r,get,lambda field:z[f[field]['key']])
    if r['mode']=='native':
        descriptor=get('descriptors').reshape(result['events'],16)
        # This qualified dense, globally learning-enabled descriptor contract
        # has sequential job IDs, 128 input bytes and no unused payload fields.
        assert np.array_equal(descriptor[:,0]>>16,np.arange(1,result['events']+1)),'descriptor job sequence'
        assert np.all((descriptor[:,0]&65535)==0x2001),'dense learning descriptor flags'
        assert np.all(descriptor[:,2]==0x10080),'dense128 input extent'
        assert not descriptor[:,[3,4,5,6,9,10,11,12,13,14,15]].any(),'reserved descriptor payload'
        assert not (descriptor[:,8]&65535).any(),'reserved descriptor label bits'
        for index in range(result['events']):
            packet=get(f'packet{index}')
            assert (int(packet[0])&0x1fffffff)==32,'captured dense128 packet extent'
    return result


def audit():
    m=json.loads(BASE.with_suffix('.json').read_text())
    assert sha(BASE.with_suffix('.npz'))==m['archive_sha256']
    rows=[];rejects=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for fixture in m['fixtures'].values():
            for v in fixture['arrays'].values():assert digest(z[v['key']])==v['sha256']
        for name,item in m['runs'].items():
            for v in item['arrays'].values():assert digest(z[v['key']])==v['sha256']
            if item['packets']:
                packets=z[item['packets']['key']];assert digest(packets)==item['packets']['sha256']
                for i,p in enumerate(packets):assert digest(p)==item['receipt']['hashes'][f'result.packet{i}']
            rows.append(dict(name=name,**check_one(z,m,name)))
        mutation_cell=run.name('native',151101,'native','pilot',0)
        for field in ['scores','inputs','states0','states1','states2','descriptors','learning_modes','packet0']:
            try:check_one(z,m,mutation_cell,field)
            except (AssertionError,ValueError,IndexError):rejects.append(field)
            else:raise AssertionError(f'Semantic mutation accepted: {field}')
        # Match actual prefixes between native and controls, not just initial seeds.
        for row in rows:
            if row['mode']!='native' or row['arm']=='native':continue
            target=run.name('native',row['seed'],'native',row['stage'],row['orientation'])
            assert target in m['runs']
            events=run.model.schedule(row['stage'],row['arm'],row['orientation'])
            end=next(e['index'] for e in events if e['block'])
            for bank in range(3):
                field=f'states{bank}'
                a=z[m['runs'][row['name']]['arrays'][field]['key']].reshape(len(events),-1)
                b=z[m['runs'][target]['arrays'][field]['key']].reshape(len(events),-1)
                assert np.array_equal(a[:end],b[:end]),'actual matched learned prefix'
    native=[r for r in rows if r['mode']=='native']
    assert {(r['seed'],r['arm'],r['stage'],r['orientation']) for r in native}==set(run.CELLS)
    assert len(native)==17 and sum(r['events'] for r in native)==11632
    assert m['state']['closed'] and not m['state'].get('failure') and not m['state'].get('owned_jobs')
    out=dict(status='NUMERIC_ARCHIVE_PASS',physical_sessions=len(native),native_forwards=sum(r['events'] for r in native),
             matched_cpu_native_cells=len(rows),semantic_mutations_rejected=rejects,actual_learned_prefixes_match=True,
             scope=m['scope'],rows=rows)
    save(BASE.with_suffix('.audit.json'),out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args()
    r=build() if a.action=='build' else audit()
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
