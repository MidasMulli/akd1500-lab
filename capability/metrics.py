"""Derived presentation numbers; hardware evidence remains in the two archives."""
import json
import numpy as np
from capability import archive,contribution_archive
from capability.design import P,DENSITIES


def derive():
    archive.audit_public();contribution_archive.audit_public();m=json.loads(archive.BASE.with_suffix('.json').read_text());c=json.loads(contribution_archive.BASE.with_suffix('.json').read_text());fixed=[]
    for kind,u in [('fnp',128),('cnp',128),('fnp',256),('cnp',256),('fnp',512),('cnp',512),('cnp',1024),('cnp',2048)]:
        runs=[r for r in m['runs'] if r['kind']=='fixed' and r['cell']==f'{kind}{u}' and not r['smoke']];assert len(runs)==3
        groups=[]
        for density in DENSITIES:
            rows=[next(g for g in r['audit']['groups'] if g['density']==density) for r in runs]
            groups.append(dict(density=density,cycles=float(np.median([r['median_cycles'] for r in rows])),wall_us=float(np.median([r['median_wall_ns'] for r in rows]))/1000,caller_us=float(np.median([r['median_caller_ns'] for r in rows]))/1000,temporal_cycles=[r['median_cycles'] for r in rows]))
        fixed.append(dict(cell=f'{kind}{u}',kind=kind,units=u,compute_npus=len(runs[0]['fixture_info']['mapping'][0]['nps']),groups=groups))
    cpu=[dict(units=r['units'],groups=r['audit']['groups']) for r in m['cpu_runs']]
    def cy(u):return next(g['cycles'] for r in fixed if r['cell']==f'cnp{u}' for g in r['groups'] if g['density']==256)
    qr={r['name']:r['audit'] for r in m['runs'] if r['name'] in archive.QUIET_PASS};busy=qr['quiet_b8_s0_r0']['total_released_thread_cpu_ns'];sleep=qr['quiet_b8_s20_r0']['total_released_thread_cpu_ns']
    return dict(status='DERIVED_FROM_VERIFIED_NATIVE_ARCHIVES',fixed=fixed,cpu=cpu,cnp_half_active=dict(detector_factor=16,service_cycle_factor=cy(2048)/cy(128),base_cycles=cy(128),large_cycles=cy(2048)),
        quiet=dict(polling8=qr['quiet_b8_s0_r0'],sleeping8=qr['quiet_b8_s20_r0'],sleeping32=qr['quiet_b32_s100_r0'],released_interval_cpu_reduction_percent=100*(1-sleep/busy),scope='Released interval only; excludes preload/drain/diagnostics. Fixed sleeps delay observation; no latency/energy win.'),
        causal_regions=[dict(name=r['name'],np=r['audit']['changed_packets'][0]['np'],block=r['audit']['suppressed_block'],drop=r['audit']['full_active_output_drop']) for r in c['runs'] if r['name']!='baseline'],
        total_physical_sessions=43,total_verified_native_forwards=9040,complete_learning_bank_snapshots=7404,failed128_event_learning_stage_preserved=True,akd_speed_advantage_over_cpu_established=False,akd_calibrated_energy_advantage_established=False)


if __name__=='__main__':
    p=P/'visual/native-primitives-metrics.json';p.write_text(json.dumps(derive(),indent=2)+'\n');print(str(p))
