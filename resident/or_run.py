"""Explicit staged admissions and independent, whole-memory resident-chain recount."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np
from resident.or_design import WIDTHS as COUNTS, SEEDS, ARMS, Network, schedule
from resident.design import gate

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261023-fnp-or-relay'
BASE = P / 'results/2026-10-23-fnp-or-relay'


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + '\n')


def freeze():
    files = list((P / 'resident').glob('*.py'))
    files += [P / p for p in ['src/or_bench.cpp', 'src/readout_bench.cpp',
        'src/mmio_driver.hpp', 'tools/build.sh', 'pong/relearn/model.py', 'build/or_bench',
        'tools/program_codec.py', 'tools/prepare_network.py', 'tools/prepare_reduction.py', 'metal/prepare.py']]
    files += list((S / 'fixtures01').iterdir())
    files += [S / 'AUTHORITY.json', S / 'PROTOCOL.md']
    p = S / 'evidence/FROZEN.json'; assert not p.exists()
    save(p, dict(status='FROZEN_BEFORE_PHYSICAL', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hashes={str(f.relative_to(P)): digest(f) for f in files if f.is_file()}))


def name_for(count, seed, arm, smoke):
    return f'{"smoke" if smoke else "full"}_u{count}_s{seed}_{arm}'


def admit(count, seed, arm, smoke):
    state = json.loads((S / 'STATE.json').read_text())
    assert not state['research_admissions_closed'] and not state.get('halted')
    assert count in COUNTS and seed in SEEDS and arm in ARMS
    assert count > 1 or arm != 'wrong_bank'
    if smoke:
        assert seed == SEEDS[0] and arm == 'repair'
        i = COUNTS.index(count)
        if i:
            prior = S / 'raw' / name_for(COUNTS[i-1], seed, 'repair', True) / 'AUDIT.json'
            assert json.loads(prior.read_text())['status'] == 'EXACT_NATIVE_CHAIN_PASS'
    else:
        assert count == 512, 'this campaign confirms only the largest qualified population'
        prior = S / 'raw' / name_for(count, SEEDS[0], 'repair', True) / 'AUDIT.json'
        assert json.loads(prior.read_text())['status'] == 'EXACT_NATIVE_CHAIN_PASS'
        if seed != SEEDS[0]:
            assert comparison(SEEDS[0])['gate']['status'] == 'PASS'
    name = name_for(count, seed, arm, smoke)
    assert name not in [r['name'] for r in state['admitted_runs']]
    entry = dict(name=name, banks=3, width=count, seed=seed, arm=arm, smoke=smoke,
        admitted_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    state['admitted_runs'].append(entry); state['stage'] = 'NATIVE_STAGED'
    save(S / 'STATE.json', state)
    return name


def run(name):
    state = json.loads((S / 'STATE.json').read_text()); auth = json.loads((S / 'AUTHORITY.json').read_text())
    assert not state['research_admissions_closed'] and not state.get('halted')
    entries = [r for r in state['admitted_runs'] if r['name'] == name]; assert len(entries) == 1
    entry = entries[0]; frozen = S / 'evidence/FROZEN.json'
    for f, h in json.loads(frozen.read_text())['hashes'].items(): assert digest(P / f) == h, ('source drift', f)
    now = dt.datetime.now(dt.timezone.utc); cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert (cutoff-now).total_seconds() > 10
    x, rows = schedule(entry['arm'], entry['smoke']); count = len(rows)
    prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
    assert len(prior) < auth['maximum_physical_sessions']
    assert sum(r['samples'] for r in prior) + count <= auth['maximum_forwards']
    f = S / 'fixtures01' / f'u{entry["width"]}_s{entry["seed"]}'
    meta = json.loads(f.with_suffix('.json').read_text());meta['banks']=3;assert len(meta['learning_nps'])==3; assert meta['status'] == 'COMPILER_VIRTUAL_ADAPTER_PASS'
    d = S / 'raw' / name
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
        d.mkdir(parents=True, exist_ok=False)
        (d / 'stream.inputs').write_bytes(x.tobytes())
        (d / 'stream.labels').write_bytes(np.asarray([r['label'] for r in rows], '<i4').tobytes())
        (d / 'stream.masks').write_bytes(bytes(r['mask'] for r in rows)); save(d / 'stream.rows.json', rows)
        rec = dict(**entry, samples=count, start_utc=now.isoformat(), source_freeze_sha256=digest(frozen), fixture_info=meta)
        save(d / 'OWNER.json', rec)
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(['taskset', '-c', '2', str(P / 'build/or_bench'), str(f), str(d/'stream'), str(d/'native'), str(count), '1'], stdout=out, stderr=err)
            rec['native_pid'] = proc.pid; save(d / 'OWNER.json', rec)
            try: code = proc.wait(timeout=min(auth['maximum_individual_seconds'], int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        rec.update(status='PASS' if code == 0 else 'FAILED_PRESERVED', exit_code=code,
            device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
            finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rec['hashes'] = {p.name: digest(p) for p in d.iterdir() if p.is_file() and p.name != 'RECEIPT.json'}
        save(d/'RECEIPT.json', rec)
        try:
            assert code == 0 and rec['device_released'], (d/'native.stderr').read_text()
            report = audit(d)
        except Exception:
            state.update(halted=True, failure=name); save(S/'STATE.json', state); raise
    print(json.dumps(report), flush=True)


def arrays(d, rec):
    c, nb = rec['samples'], rec['banks']; data = {}
    shapes = {'stream.inputs': ('u1', (c,512)), 'stream.labels': ('<i4',(c,)), 'stream.masks': ('u1',(c,)),
        'native.scores': ('<i4',(c,4)), 'native.cycles': ('<u4',(c,)), 'native.counter_bounds': ('<u4',(c,2)),
        'native.config_bounds': ('<u4',(c,2)), 'native.control_bounds': ('<u4',(c,2)), 'native.idle_counter': ('<u4',(6,)),
        'native.latency_ns': ('<i8',(c,)), 'native.caller_ns': ('<i8',(c,)), 'native.mode_ns': ('<i8',(c,)),
        'native.diagnostic_ns': ('<i8',(c,))}
    for k,(typ,shape) in shapes.items(): data[k] = np.fromfile(d/k, typ).reshape(shape)
    for k in range(nb):
        data[f'bank{k}.initial'] = np.fromfile(d/f'native.bank{k}.initial.ram','<u4')
        data[f'bank{k}.final'] = np.fromfile(d/f'native.bank{k}.ram','<u4')
        data[f'bank{k}.snapshots'] = np.stack([np.fromfile(d/f'native.snapshot_{i}_bank{k}.ram','<u4') for i in range(c)])
    for k in range(len(rec['fixture_info']['fixed_packets'])):
        for when in ['initial','final']: data[f'fixed{k}.{when}'] = np.fromfile(d/f'native.fixed{k}.{when}.ram','<u4')
    return data


def recount(rec, data):
    nb = rec['banks']; meta = rec['fixture_info']; assert meta['seed'] == rec['seed'] and meta['banks'] == nb
    x, rows = schedule(rec['arm'], rec['smoke']); assert rec['samples'] == len(rows) and rec['rows'] == rows
    for key,want in [('stream.inputs',x), ('stream.labels',[r['label'] for r in rows]), ('stream.masks',[r['mask'] for r in rows])]:
        assert np.array_equal(data[key], want), ('schedule',key)
    ram = [data[f'bank{k}.initial'].tolist() for k in range(nb)]
    for k in range(nb): assert hashlib.sha256(np.asarray(ram[k],'<u4').tobytes()).hexdigest() == meta['bank_ram_sha256'][k]
    net = Network(ram,rec['width']); edits=[]; messages=[]; frozen_start=None
    for i,(a,row) in enumerate(zip(x,rows)):
        if row['phase']=='old' and frozen_start is None: frozen_start=[b.ram[:] for b in net.banks]
        before=[b.ram[:] for b in net.banks]; y,detail=net.forward(a,row['label'],row['mask'])
        assert np.array_equal(data['native.scores'][i],y), ('output',i)
        change=[]
        for k in range(nb):
            got=data[f'bank{k}.snapshots'][i]; assert np.array_equal(got,net.banks[k].ram), ('whole RAM',i,k)
            ids=np.flatnonzero((np.asarray(before[k][2:],np.uint32).reshape(rec['width'],20)!=got[2:].reshape(rec['width'],20)).any(axis=1))
            assert row['mask']&(1<<k) or not len(ids), ('protected bank changed',i,k)
            change.append(len(ids))
        edits.append(change); messages.append(detail)
    for k in range(nb): assert np.array_equal(data[f'bank{k}.final'],net.banks[k].ram)
    for k,packet in enumerate(meta['fixed_packets']):
        assert np.array_equal(data[f'fixed{k}.initial'],data[f'fixed{k}.final']), ('fixed RAM',k)
        assert hashlib.sha256(data[f'fixed{k}.initial'].astype('<u4').tobytes()).hexdigest()==packet['sha256']
    idle=data['native.idle_counter']; bounds=data['native.counter_bounds']; ctl=data['native.control_bounds']
    assert idle[0]==idle[1]==idle[2]==idle[3] and idle[4]==idle[5], 'idle counter'
    assert np.all(data['native.config_bounds'][:,0]==data['native.config_bounds'][:,1]), 'service configuration changed'
    assert np.array_equal((bounds[:,1].astype(np.uint64)-bounds[:,0].astype(np.uint64))%2**32,data['native.cycles'])
    assert np.all(data['native.cycles']>0) and np.all(ctl==ctl[0,0]) and int(ctl[0,0])&0x300==0
    pred=data['native.scores'].argmax(axis=1);metrics={}
    for phase in ['old','changed_before','changed_after','retention']:
        ii=[i for i,r in enumerate(rows) if r['phase']==phase];correct=int(sum(pred[i]==rows[i]['destination'] for i in ii))
        metrics[phase]=dict(correct=correct,count=len(ii),accuracy=correct/len(ii))
    protected=[k for k in range(nb) if k!=(nb-1 if rec['arm']=='wrong_bank' else 0)]
    edits=np.asarray(edits); repair=[i for i,r in enumerate(rows) if r['phase']=='repair']
    return dict(status='EXACT_NATIVE_CHAIN_PASS',name=rec['name'],banks=nb,seed=rec['seed'],arm=rec['arm'],smoke=rec['smoke'],
        samples=len(rows),width=rec['width'],metrics=metrics,protected_banks_unchanged=all(net.banks[k].ram==frozen_start[k] for k in protected),
        repair_bank_edit_events=edits[repair].sum(axis=0).tolist(),all_bank_edit_events=edits.sum(axis=0).tolist(),
        median_host_ns=float(np.median(data['native.latency_ns'])),median_service_cycles=float(np.median(data['native.cycles'])),
        reconstructed_messages=messages,per_event_edited_records=edits.tolist())


def audit(d):
    rec=json.loads((d/'RECEIPT.json').read_text()); assert rec['status']=='PASS' and rec['device_released']
    for p,h in rec['hashes'].items(): assert digest(d/p)==h,p
    rec['rows']=json.loads((d/'stream.rows.json').read_text());report=recount(rec,arrays(d,rec));save(d/'AUDIT.json',report)
    return {k:v for k,v in report.items() if k not in ['reconstructed_messages','per_event_edited_records']}


def comparison(seed):
    reports={a:json.loads((S/'raw'/name_for(512,seed,a,False)/'AUDIT.json').read_text()) for a in (ARMS if seed==SEEDS[0] else ['repair','frozen'])}
    g=gate(reports['repair'],reports['frozen'])
    return dict(seed=seed,gate=g,arms={a:{k:v for k,v in r.items() if k not in ['reconstructed_messages','per_event_edited_records']} for a,r in reports.items()})


def expected_runs():
    return {name_for(c,SEEDS[0],'repair',True) for c in COUNTS}|{name_for(512,s,a,False) for s in SEEDS for a in ['repair','frozen']}|{name_for(512,SEEDS[0],'wrong_bank',False)}


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed']
    assert {d.name for d in (S/'raw').iterdir() if True}==expected_runs()
    data={};runs=[]
    for d in sorted(d for d in (S/'raw').iterdir() if True):
        audit(d);rec=json.loads((d/'RECEIPT.json').read_text());rec['rows']=json.loads((d/'stream.rows.json').read_text())
        rec['audit']=json.loads((d/'AUDIT.json').read_text());rec['receipt_sha256']=digest(d/'RECEIPT.json');rec.pop('hashes');rec.pop('native_pid')
        runs.append(rec)
        for k,v in arrays(d,rec).items():data[d.name+'__'+k]=v
    np.savez_compressed(BASE.with_suffix('.npz'),**data)
    save(BASE.with_suffix('.json'),dict(status='OBSERVED_THREE_FNP_OR_POPULATIONS',runs=runs,comparisons=[comparison(s) for s in SEEDS],
        physical_sessions=len(runs),native_forwards=sum(r['samples'] for r in runs),archive_sha256=digest(BASE.with_suffix('.npz'))))
    return audit_public()


def audit_public():
    report=json.loads(BASE.with_suffix('.json').read_text()); assert digest(BASE.with_suffix('.npz'))==report['archive_sha256']
    assert {r['name'] for r in report['runs']}==expected_runs()
    assert report['physical_sessions']==len(report['runs']) and report['native_forwards']==sum(r['samples'] for r in report['runs'])
    checked=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for rec in report['runs']:
            result=recount(rec,{k.split('__',1)[1]:z[k] for k in z.files if k.startswith(rec['name']+'__')})
            assert result==rec['audit'];checked.append(result)
    for seed in SEEDS:
        rr={r['arm']:r for r in checked if r['seed']==seed and not r['smoke']}
        assert gate(rr['repair'],rr['frozen'])['status']=='PASS'
    return dict(status='PASS',physical_sessions=len(checked),native_forwards=report['native_forwards'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('args',nargs='*');a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='admit':print(admit(int(a.args[0]),int(a.args[1]),a.args[2],a.args[3]=='smoke'))
    elif a.command=='run':run(a.args[0])
    elif a.command=='audit':print(json.dumps(audit(S/'raw'/a.args[0])))
    elif a.command=='compare':print(json.dumps(comparison(int(a.args[0]))))
    elif a.command=='export':print(json.dumps(export()))
    elif a.command=='audit-public':print(json.dumps(audit_public()))
    else:raise ValueError(a.command)
