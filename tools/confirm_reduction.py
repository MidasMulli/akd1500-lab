"""Three prospectively gated interleaved mask/reduced timing pairs."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
from verify_reduction_pair import check
P=Path(__file__).resolve().parents[1];S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-class-reduction')

def main():
    gate=json.loads((S/'evidence/LARGE_PAIR.json').read_text());assert gate['status']=='PASS' and gate['samples']==64
    (S/'evidence/TIMING_ADMISSION.json').write_text(json.dumps(dict(stage='TIMING_AFTER_INDEPENDENT_LARGE_PAIR',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pairs=3,samples=256,cpu_repeats=256,binary_sha256=hashlib.sha256((P/'build/readout_bench').read_bytes()).hexdigest()),indent=2)+'\n')
    for r in range(3):
        names={}
        for contract in ['mask','reduced'] if r%2==0 else ['reduced','mask']:
            name=f'confirm_{r}_{contract}';names[contract]=name
            family='20261023-compact-readout' if contract=='mask' else S.name
            fixture=f'studies/{family}/fixtures/n512_u512_{contract}'
            subprocess.run([sys.executable,str(P/'tools/run_cell.py'),name,fixture,'--samples','256','--cpu-repeats','256','--bench','readout_bench'],check=True)
        report=check(P,S/'raw'/names['mask'],S/'raw'/names['reduced']);(S/'evidence'/f'PAIR_{r}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(stage='PAIR_VERIFIED',rep=r,exact_counts=report['exact_group_counts'],full_RAM_words_equal=report['complete_RAM_words_equal'])),flush=True)
    print('PASS: three independently recounted native-reduction pairs')

if __name__=='__main__':main()
