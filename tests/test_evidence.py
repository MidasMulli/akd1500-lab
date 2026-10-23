from pathlib import Path
import json,unittest
P=Path(__file__).resolve().parents[1]
class EvidenceTests(unittest.TestCase):
    def test_public_raw_replay_matches_visible_edits(self):
        try:import numpy as np
        except ImportError:self.skipTest('Install numpy to verify the public raw archive')
        archive=P/'results/mesh32-evidence.npz'
        if not archive.exists():self.skipTest('Optional measured evidence archive not generated')
        data=np.load(archive,allow_pickle=False);replay=json.loads((P/'visual/mesh32-data.json').read_text());before=data['initial_ram']
        self.assertEqual(data['measured_scores'].shape,(64,64));self.assertEqual(data['measured_ram_after'].shape,(64,514))
        total=0
        for i,after in enumerate(data['measured_ram_after']):
            edits=[]
            self.assertEqual(data['measured_scores'][i].tolist(),replay['samples'][i]['scores'])
            for j in range(64):
                old={b for b in range(128) if int(before[2+j*8+4+b//32])>>(b%32)&1};new={b for b in range(128) if int(after[2+j*8+4+b//32])>>(b%32)&1}
                self.assertEqual(len(old),32);self.assertEqual(len(new),32)
                if old!=new:edits.append(dict(neuron=j,removed=sorted(old-new),added=sorted(new-old)));total+=len(new-old)
            self.assertEqual(edits,replay['samples'][i]['connection_edits']);before=after
        self.assertEqual(total,1004)
    def test_placement_is_thirty_one_fixed_plus_one_learner(self):
        f=P/'visual/mesh32-data.json'
        if not f.exists():self.skipTest('Optional mesh evidence not generated')
        m=json.loads(f.read_text())['mapping'];layers=m['layer_assignments'];self.assertEqual(len(layers),32)
        self.assertEqual(len({tuple(c['np']) for l in layers for c in l['components']}),32)
        self.assertEqual(sum(c['type']=='Type.FNP3' for l in layers for c in l['components']),1)
if __name__=='__main__':unittest.main()
