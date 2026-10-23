"""Use the fixed compiler, then remove every pre-training test query explicitly."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
from bank import save, sha


def build(train,test,output,lock):
    intermediate=output.with_name(output.name+'-UNADMITTED-PREQUERIES')
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare.py')),str(train),str(test),str(intermediate),'--phase','test','--selected',str(lock)],check=True)
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for row in json.loads((intermediate/'BUILD.json').read_text())['rows']:
        source=intermediate/row['prefix'];target=output/row['prefix'];q=row['evaluation_samples']
        n=int(source.with_suffix('.meta').read_text().split()[0])
        for suffix in ['bin','ram','meta']:shutil.copy2(source.with_suffix('.'+suffix),target.with_suffix('.'+suffix))
        target.with_suffix('.inputs').write_bytes(source.with_suffix('.inputs').read_bytes()[q*n:])
        target.with_suffix('.labels').write_bytes(source.with_suffix('.labels').read_bytes()[q*4:])
        meta=json.loads(source.with_suffix('.json').read_text());meta.update(samples=meta['samples']-q,prequeries=0,regime='INDUCTIVE_TRAIN_THEN_TEST')
        save(target.with_suffix('.json'),meta);rows.append(dict(row,samples=meta['samples'],prequeries=0))
    save(output/'BUILD.json',dict(status='VIRTUAL_ONLY_PASS',phase='confirmation',prequeries=0,rows=rows,source_sha256=sha(__file__),physical_device_opened=False))
    print('INDUCTIVE_VIRTUAL_PASS',len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('train',type=Path);p.add_argument('test',type=Path);p.add_argument('output',type=Path);p.add_argument('lock',type=Path);a=p.parse_args();build(a.train.resolve(),a.test.resolve(),a.output.resolve(),a.lock.resolve())
