"""Gated, interleaved repeats for the compact-output study."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
from verify_readout_pair import check
P=Path(__file__).resolve().parents[1]
S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-compact-readout')

def main():
    gate=json.loads((S/'evidence/LARGE_PAIR.json').read_text())
    assert gate['status']=='PASS' and gate['samples']==64 and gate['neurons']==512
    assert gate['true_responses']>0 and gate['false_responses']>0
    admitted=dict(stage='TIMING_AFTER_INDEPENDENT_LARGE_PAIR',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pairs=6,samples=256,cpu_repeats=256,modes=['serial','resident'],contract_order='Alternates across reps and queue modes',binary_sha256=hashlib.sha256((P/'build/readout_bench').read_bytes()).hexdigest())
    (S/'evidence/TIMING_ADMISSION.json').write_text(json.dumps(admitted,indent=2)+'\n')
    for r in range(3):
        for mode in ['serial','resident'] if r%2==0 else ['resident','serial']:
            names={}
            for contract in ['potentials','mask'] if (r+(mode=='resident'))%2==0 else ['mask','potentials']:
                name=f'confirm_{r}_{mode}_{contract}';names[contract]=name
                fixture=str((S/'fixtures'/f'n512_u512_{contract}').relative_to(P))
                args=[sys.executable,str(P/'tools/run_cell.py'),name,fixture,'--samples','256','--cpu-repeats','256','--bench','readout_bench']
                if mode=='resident':args+=['--resident']
                subprocess.run(args,check=True)
            report=check(P,S/'raw'/names['potentials'],S/'raw'/names['mask'])
            (S/'evidence'/f'PAIR_{r}_{mode}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(stage='PAIR_VERIFIED',rep=r,mode=mode,threshold_responses=report['exact_threshold_responses'],learned_words_equal=report['learned_words_equal'])),flush=True)
    print('PASS: six independently recounted pairs')

if __name__=='__main__':main()
