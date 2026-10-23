"""Exact physical recount and defenses against substituted decisions or RGB sources."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mario.sealed.archive import materialize
from mario.continuous.model import Reference
from mario.continuous.audit_public import audit
from verify import verify


class ContinuousEvidence(unittest.TestCase):
    def test_every_physical_response_and_RAM_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=materialize(root,stem='2026-10-23-mario-continuous')
            self.assertEqual(len(runs),3);self.assertEqual(sum(verify(root,r)['samples'] for r in runs),3385)
            f=root/'raw/dev_full01';z=root/'raw/dev_frozen01'
            self.assertEqual((root/'fixtures/dev_full01.ram').read_bytes(),(root/'fixtures/dev_frozen01.ram').read_bytes())
            self.assertEqual((z/'native.ram').read_bytes(),(root/'fixtures/dev_frozen01.ram').read_bytes())
            self.assertNotEqual((f/'native.ram').read_bytes(),(root/'fixtures/dev_full01.ram').read_bytes())

    def test_stateful_controller_reference_matches_earlier_physical_stream(self):
        z=np.load(ROOT/'results/2026-10-23-mario-confirmation.npz',allow_pickle=False);n='seed82911_full';model=Reference(z[f'fixture__{n}__ram'])
        x=z[f'fixture__{n}__inputs'].reshape(-1,1024);labels=z[f'fixture__{n}__labels'];scores=z[n+'__native_scores'].reshape(-1,192);states=z[n+'__snapshots']
        for i,(v,label) in enumerate(zip(x,labels)):
            self.assertEqual(model.forward(v,int(label)),scores[i].tolist());self.assertEqual(model.ram,states[i].tolist())

    def test_all_live_decisions_feedback_and_game_sources_bound(self):
        self.assertEqual(audit(),dict(status='PUBLIC_CONTINUOUS_BINDING_PASS',decisions=115,bound_responses=345,frames=2646))

    def test_substituted_action_rejected(self):
        data=json.loads((ROOT/'visual/mario-continuous-data.json').read_text());data=copy.deepcopy(data)
        decision=next(e for e in data['captures']['full']['events'] if e['kind']=='decision');decision['action']^=1
        with self.assertRaisesRegex(AssertionError,'action does not follow native replies'):audit(data)

    def test_reassigned_query_frame_rejected(self):
        data=json.loads((ROOT/'visual/mario-continuous-data.json').read_text());data=copy.deepcopy(data)
        event=next(e for e in data['captures']['full']['events'] if e['kind']=='forward');event['input_frame']+=24
        with self.assertRaises(AssertionError):audit(data)

    def test_ended_video_arm_never_displays_its_later_episode_responses(self):
        from tools.visualize_mario_continuous import arm_wall
        data=json.loads((ROOT/'visual/mario-continuous-data.json').read_text())['captures']['full']
        events=data['events'];start=next(e['mono_ns'] for e in events if e['kind']=='frame' and e['episode']==2)
        end=next(e['mono_ns'] for e in events if e['kind']=='episode_end' and e['episode']==2)
        self.assertLess(end-start,8_000_000_000)
        selected=arm_wall(start,end,8.0)
        packets=[e for e in events if e['kind']=='forward' and e['response']['received_monotonic_ns']<=selected]
        source=next(e for e in events if e['kind']=='frame' and e['frame_id']==packets[-1]['input_frame'])
        self.assertEqual(source['episode'],2)
        self.assertTrue(any(e['kind']=='forward' and end<e['response']['received_monotonic_ns']<start+8_000_000_000 for e in events))

if __name__=='__main__':unittest.main()
