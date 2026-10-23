"""Reconstruct published raw native evidence; reject mutually consistent bad data."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from verify import verify
from verify_network import verify as verify_network

def materialize(root,label):
    z=json.loads((P/f'results/2026-10-23-{label}.json').read_text());archive=P/f'results/2026-10-23-{label}.npz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==z['archive_sha256']
    data=np.load(archive,allow_pickle=False);network=label in ['network','balanced-network'];paths=[]
    for r in z['runs']:
        if r['status']!='PASS' or 'fixture_key' not in r:continue
        key=r['fixture_key'];meta=z['fixtures'][key];f=root/'fixtures'/key;f.parent.mkdir(exist_ok=True)
        prefix='fixture__'+key+'__'
        for k in data.files:
            if k.startswith(prefix):
                ext=k[len(prefix):].replace('_ram','.ram')
                if ext.startswith('bank') or ext.startswith('fixed'):suffix='.'+ext
                else:suffix='.'+ext
                Path(str(f)+suffix).write_bytes(data[k].astype('u1' if ext=='inputs' else '<u4').tobytes())
        d=root/'runs'/r['name'];d.mkdir(parents=True)
        for k in data.files:
            if k.startswith(r['name']+'__'):
                ext=k[len(r['name'])+2:]
                if ext.endswith('_ram'):name=ext[:-4].replace('native_','native.',1).replace('_initial','.initial').replace('_final','.final')+'.ram'
                elif ext.endswith('_scores'):name=ext[:-7]+'.scores'
                else:continue
                (d/name).write_bytes(data[k].astype('<u4').tobytes())
        rec=dict(status='PASS',device_released=True,fixture=str(f.relative_to(root)),samples=r['samples'],hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in d.iterdir()})
        if network:
            nf=sum(k.startswith(prefix+'fixed') for k in data.files);plan=dict(meta,fixed=[{}]*nf)
            f.with_suffix('.json').write_text(json.dumps(plan));rec.update(learning_mask=r['learning_mask'],fixture_hashes={})
        else:
            f.with_suffix('.meta').write_text(f'{meta["inputs"]} {meta["neurons"]} {meta["num_weights"]} {meta["classes"]}')
            rec.update(learning=r['learning'],activated=bool(meta['output_contract']),output_contract=meta['output_contract'],threshold=meta['threshold'])
        (d/'RECEIPT.json').write_text(json.dumps(rec));paths.append(d)
    return paths

class ExtensionEvidence(unittest.TestCase):
    def test_all_offload_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=materialize(root,'offload');self.assertEqual(len(runs),17)
            for d in runs:self.assertEqual(verify(root,d)['status'],'PASS')
    def test_all_network_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=materialize(root,'network');self.assertEqual(len(runs),7)
            for d in runs:self.assertEqual(verify_network(root,d)['status'],'PASS')
    def test_balanced_network_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=materialize(root,"balanced-network");self.assertEqual(len(runs),6)
            for d in runs:self.assertEqual(verify_network(root,d)["status"],"PASS")
    def test_corrupted_network_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=materialize(root,'network')[-1];p=d/'native.scores';b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b)
            r=json.loads((d/'RECEIPT.json').read_text());r['hashes'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();(d/'RECEIPT.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(AssertionError,'final output mismatch'):verify_network(root,d)
    def test_corrupted_network_RAM_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=materialize(root,'network')[-1];p=d/'native.snapshot_0_bank1.ram';b=bytearray(p.read_bytes());b[28]^=1;p.write_bytes(b)
            r=json.loads((d/'RECEIPT.json').read_text());r['hashes'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();(d/'RECEIPT.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(AssertionError,'bank snapshot mismatch'):verify_network(root,d)
if __name__=='__main__':unittest.main()
