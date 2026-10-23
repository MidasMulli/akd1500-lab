import importlib.util,math,sys,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'power'))
from ina228 import INA228
from analyze import analyze
class I2C:
    def __init__(self):self.regs={0x3e:0x5449,0x3f:0x2281,0xb:0,5:16896<<4,7:10000<<4,8:10312,9:100,4:4800<<4};self.writes=[]
    def readfrom_mem(self,a,r,n):return self.regs[r].to_bytes(n,'big')
    def writeto_mem(self,a,r,b):self.regs[r]=int.from_bytes(b,'big');self.writes.append(r)
class PowerTests(unittest.TestCase):
    def test_register_units_and_readback(self):
        i=I2C();s=INA228(i);v=s.sample();self.assertAlmostEqual(v['bus_v'],3.3);self.assertAlmostEqual(v['current_a'],.1);self.assertAlmostEqual(v['energy_j'],.0512);self.assertEqual(s.cal,1966)
    def test_identity_failure_writes_nothing(self):
        i=I2C();i.regs[0x3e]=0
        with self.assertRaises(RuntimeError):INA228(i)
        self.assertEqual(i.writes,[])
    def test_overflow_fails(self):
        i=I2C();s=INA228(i);i.regs[0xb]=1<<11
        with self.assertRaises(RuntimeError):s.sample()
    def fixture(self):
        h=dict(kind='recorder',source='ina228_hardware',synthetic=True,calibration_id='synthetic',measurement_domain='akd_card_input',current_lsb_a=1e-5)
        rows=[dict(record=h)]
        for i in range(101):rows.append(dict(record=dict(kind='sample',sequence=i,sensor_elapsed_us=i*10000,bus_v=3.3,current_a=.1,power_w=.33,energy_j=round(.33*i*.01/.000512)*.000512,diag_flags=0)))
        c=dict(id='synthetic',known_load_check_passed=True,zero_check_passed=True,all_card_supply_paths_intercepted=True,relative_uncertainty=.01)
        return rows,c
    def test_synthetic_integration_and_physical_rejection(self):
        rows,c=self.fixture();self.assertEqual(analyze(rows,c,True)['status'],'SYNTHETIC_PIPELINE_TEST')
        with self.assertRaises(AssertionError):analyze(rows,c,False)
    def test_missing_sample_and_uncalibrated_rejected(self):
        rows,c=self.fixture();del rows[20]
        with self.assertRaises(AssertionError):analyze(rows,c,True)
        rows,c=self.fixture();c['known_load_check_passed']=False
        with self.assertRaises(AssertionError):analyze(rows,c,True)
if __name__=='__main__':unittest.main()
