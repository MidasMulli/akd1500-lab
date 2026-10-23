"""Energy from a validated sensor stream; refuses uncalibrated hardware claims."""
from pathlib import Path
import argparse,json,math
def analyze(rows,calibration,synthetic=False):
    h=rows[0]['record'];assert h['kind']=='recorder' and h['source']=='ina228_hardware'
    assert h.get('synthetic') is synthetic,'Synthetic fixture must never be admitted as physical data'
    assert h['calibration_id']==calibration['id']
    assert calibration['known_load_check_passed'] and calibration['zero_check_passed']
    assert calibration['all_card_supply_paths_intercepted']
    assert math.isfinite(calibration['relative_uncertainty']) and 0<calibration['relative_uncertainty']<1
    samples=[x['record'] for x in rows if x['record']['kind']=='sample'];assert len(samples)>=10
    for i,r in enumerate(samples):
        assert r['sequence']==samples[0]['sequence']+i
        assert all(math.isfinite(r[k]) for k in ['bus_v','current_a','power_w','energy_j'])
        assert r['bus_v']>0 and r['energy_j']>=0 and not r['diag_flags']&0xe00
        if i:
            prev=samples[i-1];assert 0<r['sensor_elapsed_us']-prev['sensor_elapsed_us']<=100000
            assert r['energy_j']>=prev['energy_j'],'counter reset or wrap'
    elapsed=(samples[-1]['sensor_elapsed_us']-samples[0]['sensor_elapsed_us'])/1e6
    energy=samples[-1]['energy_j']-samples[0]['energy_j']
    integrated=sum((a['power_w']+b['power_w'])*.5*(b['sensor_elapsed_us']-a['sensor_elapsed_us'])/1e6 for a,b in zip(samples,samples[1:]))
    quantum=16*3.2*h['current_lsb_a'];assert abs(integrated-energy)<=max(2*quantum,.05*abs(energy)),'power/counter integration disagree'
    return dict(status='SYNTHETIC_PIPELINE_TEST' if synthetic else 'VALIDATED_SENSOR_INTERVAL',
                measurement_domain=h['measurement_domain'],seconds=elapsed,gross_joules=energy,
                mean_watts=energy/elapsed,trapezoid_joules=integrated,energy_counter_quantum_j=quantum,
                relative_calibration_uncertainty=calibration['relative_uncertainty'],
                endpoint='Whole interval; not an individual inference or core-only energy',
                phase_sync='USB arrival times are not precise hardware phase triggers; bracket long phases or add a separately validated GPIO trigger')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stream',type=Path);p.add_argument('calibration',type=Path);p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    print(json.dumps(analyze([json.loads(x) for x in a.stream.read_text().splitlines()],json.loads(a.calibration.read_text()),a.synthetic),indent=2))
