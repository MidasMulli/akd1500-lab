import json
import unittest
import numpy as np
from admission import learned_policy as model
from admission.learned_policy_run import fixture
from instruments.explorer.data import export, records
from instruments.explorer.live import Stepper, action_rows, validate_action


class ExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=export()

    def test_measured_deltas_reconstruct_every_capture(self):
        for arm in model.ARMS:
            s=Stepper(fixture(151101,arm),arm)
            state=json.loads(json.dumps(self.data['arms'][arm]['initial']))
            for i,e in enumerate(model.schedule('pilot',arm,0)):
                expected,_,_,rams=s.forward(e)
                shown=self.data['arms'][arm]['events'][i]
                self.assertEqual(shown['changes'],expected['changes'])
                self.assertEqual(shown['policy_flags'],expected['policy_flags'])
                self.assertEqual(shown['memory_flags'],expected['memory_flags'])
                for change in shown['changes']:
                    k,j=change['bank'],change['record'];self.assertEqual(state[k][j],change['before'])
                    state[k][j]=change['after']
                for k,ram in enumerate(rams):self.assertEqual(state[k],records(ram,model.DIMENSIONS[k][0]))

    def test_runtime_queries_never_teach(self):
        s=Stepper(fixture(151101,'native'),'native')
        for e in model.schedule('support','native'):s.forward(e)
        before=[r.ram[:] for r in s.refs]
        for e in action_rows(dict(action='query',key=0),'native',84):
            event,*_=s.forward(e);self.assertFalse(event['changes'])
        self.assertEqual(before,[r.ram for r in s.refs])

    def test_request_boundaries(self):
        invalid=[dict(action='query',key=True),dict(action='query',key=6),
                 dict(action='start',arm='native',mode='shell'),
                 dict(action='feedback',policy=4,candidate=0,outcome=1),
                 dict(action='query',key=0,path='/dev/akd1500_0'),
                 dict(action='propose',key=0,policy=-1,candidate=0)]
        for request in invalid:
            with self.assertRaises(ValueError):validate_action(request)

    def test_all_control_clocks_preserved(self):
        for arm in model.ARMS:
            rows=action_rows(dict(action='feedback',policy=0,candidate=0,outcome=1),arm,84)
            self.assertTrue(all(e['labels'][1:]==[-1,-1] for e in rows))
            if arm=='manager_frozen':self.assertTrue(all(e['labels'][0]==-1 for e in rows))
            rows=action_rows(dict(action='propose',key=0,policy=0,candidate=1),arm,90)
            if arm=='students_frozen':self.assertTrue(all(e['labels']==[-1,-1,-1] for e in rows))

    def test_evidence_boundaries_and_counts(self):
        self.assertEqual(self.data['audit']['independently_checked_streams'],10)
        self.assertEqual(sum(len(b['plastic']) for b in self.data['banks']),32)
        self.assertEqual(sum(b['units'] for b in self.data['banks']),256)
        self.assertEqual(len(self.data['mapping']),7)
        self.assertIn('Reconstructed',self.data['evidence']['features'])
        self.assertIn('not a die',self.data['evidence']['geometry'])
        for arm in self.data['arms'].values():
            self.assertEqual(len(arm['events']),268)
            for e in arm['events']:
                if e['role'] in ['query','exam']:self.assertFalse(e['changes'])
                for change in e['changes']:
                    self.assertEqual(len(change['added']),len(change['removed']))
                    self.assertEqual(len(change['after']['connections']),32)


if __name__=='__main__':unittest.main()
