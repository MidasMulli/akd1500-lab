"""Public compact-readout evidence can be independently checked without hardware."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from verify import verify

class ReadoutEvidenceTests(unittest.TestCase):
    def load(self):
        try:import numpy as np
        except ImportError:self.skipTest('numpy required for measured archive')
        path=P/'results/2026-10-23-readout.json'
        if not path.exists():self.skipTest('Optional measured readout archive absent')
        report=json.loads(path.read_text());archive=path.with_suffix('.npz')
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),report['archive_sha256'])
        return report,np.load(archive,allow_pickle=False)

    def materialize(self,root,report,data,row,corrupt=False):
        key=row['fixture_key'];meta=report['fixtures'][key];f=root/'fixtures'/key;f.parent.mkdir(exist_ok=True)
        f.with_suffix('.meta').write_text(f"{meta['inputs']} {meta['neurons']} {meta['num_weights']} {meta['classes']}\n")
        for ext in ['inputs','labels','ram']:f.with_suffix('.'+ext).write_bytes(data[key+'__'+ext].tobytes())
        run=root/'raw'/row['study']/row['name'];run.mkdir(parents=True)
        for mode in ['native','cpu']:
            for ext in ['scores','ram']:
                a=data[row['study']+'__'+row['name']+'__'+mode+'_'+ext].copy()
                if corrupt and ext=='scores':a[0]+=1
                (run/f'{mode}.{ext}').write_bytes(a.tobytes())
        receipt=dict(status='PASS',device_released=True,fixture=str(f.relative_to(root)),samples=row['samples'],learning=row['learning'],activated=bool(row['output_contract']),output_contract=row['output_contract'],threshold=row['threshold'],hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in run.iterdir()})
        (run/'RECEIPT.json').write_text(json.dumps(receipt));return run

    def test_all_public_streams_match_independent_selector(self):
        report,data=self.load();self.assertEqual(len(report['runs']),28)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for row in report['runs']:
                self.assertEqual(row['status'],'PASS');run=self.materialize(root,report,data,row)
                self.assertEqual(verify(root,run)['status'],'PASS')

    def test_matching_but_wrong_native_and_cpu_counts_are_rejected(self):
        report,data=self.load();row=next(r for r in report['runs'] if r['output_contract']==2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.materialize(root,report,data,row,corrupt=True)
            with self.assertRaisesRegex(AssertionError,'independent output mismatch'):verify(root,run)

if __name__=='__main__':unittest.main()
