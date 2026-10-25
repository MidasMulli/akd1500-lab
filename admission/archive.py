"""Public numeric-only evidence; replay both the prerequisite and final circuit."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from admission import run as fixed
from admission import learned_run as learned
from admission.learned import SEEDS
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-learned-admission'
FAMILIES={'fixed':fixed,'learned':learned}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def save(p,a):Path(p).write_text(json.dumps(a,indent=2)+'\n')

def build():
    arrays={};families={}
    for family,mod in FAMILIES.items():
        fixtures={};runs={}
        for f in sorted(mod.F.glob('s*')):
            fs={}
            for p in sorted([*f.glob('bank*.ram'),*f.glob('bank*.threshold'),*f.glob('fixed*.ram')]):
                key=family+'__'+f.name+'__'+p.name.replace('.','_');arrays[key]=np.fromfile(p,'<u4');fs[p.name]=dict(key=key,sha256=sha(p))
            fixtures[f.name]=dict(arrays=fs,metadata=json.loads((f/'BUILD.json').read_text()))
        for d in sorted((mod.S/'raw').iterdir()):
            if not (d/'RECEIPT.json').exists():continue
            fs={}
            for p in sorted(d.glob('result.*')):
                name=p.name[7:];t='u1' if name in ['inputs','scores'] else '<i4' if name.startswith('potentials') or name=='learning_modes' else '<u4';key=family+'__'+d.name+'__'+name.replace('.','_');arrays[key]=np.fromfile(p,t);fs[name]=dict(key=key,sha256=sha(p))
            raw=(d/'stdout').read_text();runs[d.name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,arrays=fs,stdout=json.loads(raw) if raw.startswith('{') else raw,stderr=(d/'stderr').read_text(),source_freeze=json.loads((d/'FROZEN.json').read_text()))
        attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in sorted(mod.S.glob('fixtures*/*/BUILD.json')) if p.parent.parent!=mod.F]
        families[family]=dict(fixtures=fixtures,runs=runs,compiler_attempts=attempts,closed_state=json.loads((mod.S/'STATE.json').read_text()))
        rejected=mod.S/'failed_attempts/cpu01'
        if (rejected/'screen.jsonl').exists():families[family]['rejected_cpu_design']=dict(status='REJECTED_CPU_FORECAST_BEFORE_HARDWARE',source_sha256=sha(rejected/'model.py'),rows=[json.loads(x) for x in (rejected/'screen.jsonl').read_text().splitlines()],reason='Initial fixed query gate did not require the current key and returned ambiguous histories. Twenty forecasts preserved; not physical results.')
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),families=families,field_provenance=dict(native_measured='captured inputs, returned output packets, all per-boundary learning-bank RAM, initial/final fixed RAM, descriptor/address captures',derived='potentials are packed CPU predictions; intermediate fixed/helper activation from own matrices and captured inputs; CPU dictionary controls are software replay',host_journal='learning_modes, OWNER/RECEIPT and program/write/cleanup counters are instrument/source journals, not electrical bus or spike traces'),scope='Seven sequential FNP3s, three resident plastic banks: one plastic recency record plus two eight-record consequence banks. Native synaptic match controls admission. Designed repeat rule and feature coding; host supplies current raw key, fixed phase clocks and delayed actual outcomes. CPU controls also succeed; no energy or speed advantage, learned utility objective, autonomous scheduler, power-loss persistence or global novelty claim. Prerequisite six-group fixed-policy study is separate.'))
    return audit()

def read_run(z,family,name):
    m=json.loads(BASE.with_suffix('.json').read_text());r=m['families'][family]['runs'][name];f=m['families'][family]['fixtures'][f"s{r['receipt']['seed']}_{r['receipt']['arm']}"]
    return r,lambda k:z[r['arrays'][k]['key']],lambda k:z[f['arrays'][k]['key']]

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];allrows={};failures=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for family,data in m['families'].items():
            mod=FAMILIES[family];rows=[]
            for bank in [*data['fixtures'].values(),*data['runs'].values()]:
                for v in bank['arrays'].values():assert digest(z[v['key']])==v['sha256']
            for name,run in data['runs'].items():
                if not run['verified']:
                    failures.append(dict(family=family,name=name,exit_code=run['receipt']['exit_code'],stderr=run['stderr'],captured_forwards=len(z[run['arrays']['scores']['key']])//128 if 'scores' in run['arrays'] else 0));continue
                r,get,fg=read_run(z,family,name);a=mod.check(r['receipt'],get,fg);a['name']=name;rows.append(a)
                if r['receipt']['mode']=='native':
                    o=r['stdout'];assert o['program_calls']==1 and o['weight_restore_calls']==o['host_bank_decisions']==0 and o['cleanup']=='unprogrammed'
            allrows[family]=rows
        main={ (q['seed'],q['arm'],q['stage']):q for q in allrows['learned'] if q['mode']=='native'}
        required={(SEEDS[0],a,'gate') for a in ['native','open','closed']}|{(SEEDS[0],'native','prefix')}|{(SEEDS[0],a,'pressure') for a in ['native','open','frozen']}|{(s,a,'heldout') for s in SEEDS for a in ['native','open','frozen']}|{(SEEDS[0],a,'heldout') for a in ['manager_frozen','severed']}
        assert set(main)==required and len(main)==24,'complete prospectively sealed physical series'
        data=m['families']['learned']
        def arr(seed,arm,stage,key):return z[data['runs'][f'native_s{seed}_{arm}_{stage}']['arrays'][key]['key']]
        for seed in SEEDS:
            for arm in ['open','frozen']:
                # Consequence-bank controls do not change the native manager trajectory.
                assert np.array_equal(arr(seed,'native','heldout','states0'),arr(seed,arm,'heldout','states0'))
            for arm in ['native','open','frozen']:
                if seed==SEEDS[0]:
                    for k,words in [('states0',386),('states1',1730),('states2',1154),('scores',128)]:assert np.array_equal(arr(seed,arm,'pressure',k),arr(seed,arm,'heldout',k)[:300*words]),'unchanged sealed prefix'
        assert np.array_equal(arr(SEEDS[0],'native','heldout','states0'),arr(SEEDS[0],'severed','heldout','states0')),'permission ablation must leave manager learning identical'
        primary={}
        for arm in ['native','open','frozen']:
            rs=[main[s,arm,'heldout'] for s in SEEDS];primary[arm]=dict(final_probe_correct=sum(r['assessment']['final_probe']['protected']['correct'] for r in rs),final_probe_total=120,heldout_correct=sum(r['assessment']['heldout']['protected']['correct'] for r in rs),heldout_total=480,new_useful_online_correct=sum(r['assessment']['new_useful']['protected']['correct'] for r in rs),new_useful_online_total=320,all_queries_correct=sum(sum(v['protected']['correct'] for v in r['assessment'].values()) for r in rs),all_queries_total=1300)
        cpu={}
        for name in ['unbounded','repeat_lru','lru']:
            rs=[main[s,'native','heldout'] for s in SEEDS];cpu[name]=dict(heldout_correct=sum(r['assessment']['heldout']['cpu'][name]['correct'] for r in rs),heldout_total=480,all_queries_correct=sum(sum(v['cpu'][name]['correct'] for v in r['assessment'].values()) for r in rs),all_queries_total=1300)
        ablations={arm:main[SEEDS[0],arm,'heldout']['assessment']['heldout']['protected'] for arm in ['manager_frozen','severed']}
    totals={family:dict(qualified_physical_sessions=sum(r['mode']=='native' for r in rows),verified_forwards=sum(r['events'] for r in rows if r['mode']=='native')) for family,rows in allrows.items()}
    result=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],totals=totals,primary=primary,cpu=cpu,first_seed_ablations=ablations,retained_original_record_ids=dict(correct=sum(main[s,'native','heldout']['old_records_unchanged'] for s in SEEDS),total=10),preserved_failures=failures,compiler_attempts=m['families']['fixed']['compiler_attempts'],rows=allrows,scope=m['scope']);save(BASE.with_suffix('.audit.json'),result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k not in ['rows','compiler_attempts']}))
