"""Meaningful controls for exploration and its learning dependence; no hardware."""
import importlib.util,pathlib,sys,unittest
P=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'image_service/service'))
from controller import CPUController
spec=importlib.util.spec_from_file_location('forecast_recovery',P/'tools/forecast_recovery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RecoveryTests(unittest.TestCase):
    def test_probe_overrides_a_known_good_action_using_last_experience(self):
        c=CPUController();c.observe(3,0,1);c.observe(1,0,0)
        self.assertEqual(c.choose(3)['action'],0)
        self.assertEqual(c.choose(3,probe=True)['action'],1)
        c.observe(3,1,3)
        self.assertEqual(c.choose(3,probe=True)['action'],0)
    def test_forecast_controls_and_charge_failed_probes(self):
        for seed in [71011,71012,71013]:
            p=module.forecast(seed,'probe');r=module.forecast(seed,'reactive');f=module.forecast(seed,'frozen')
            self.assertEqual(p['completed'],24)
            self.assertLess(f['completed'],p['completed'])
            self.assertEqual(p['failures'],sum(not e['ok'] for e in p['events']))
            self.assertEqual(p['attempts'],len(p['events']))
            self.assertTrue(any(e['job'] in range(12,16) and e['state']==3 and e['action']==0 and e['ok'] and e['probe'] for e in p['events']))
            self.assertFalse(any(e['job'] in range(12,20) and e['state']==3 and e['action']==0 for e in r['events']))
    def test_cpu_matches_probe_trajectory(self):
        for seed in [71011,71012,71013]:
            self.assertEqual(module.forecast(seed,'probe')['events'],module.forecast(seed,'cpu')['events'])

if __name__=='__main__':unittest.main()
