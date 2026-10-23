"""Frozen interleaved cells; each subprocess must pass before the next begins."""
from pathlib import Path
import json,subprocess,sys,os
P=Path(__file__).resolve().parents[1];S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-measurement')
cells=['n128_u8','n512_u64','n512_u512']
admission=dict(stage='timing_confirmation',prerequisite='All five staged FNP3 kernel pilots passed exact scores and final RAM; all four mesh gates passed',cells=cells,pairs_per_cell=3,samples=256,cpu_repeats=256,source_hashes={})
import hashlib
admission['source_hashes']={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'src/learning_bench.cpp',P/'src/mmio_driver.hpp',P/'build/learning_bench']}
(S/'evidence/CONFIRM_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
for r in range(3):
    for cell in cells if r%2==0 else list(reversed(cells)):
        subprocess.run([sys.executable,str(P/'tools/run_cell.py'),f'confirm_{r}_{cell}',f'studies/20261023-measurement/fixtures_v2/{cell}','--samples','256','--cpu-repeats','256'],check=True)
print('PASS: nine paired cells')
