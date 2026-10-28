from media_binding import binding_digest
"""Measured primitive checks, blocked-host evidence and the preserved failure."""
import copy
import hashlib
import json
import unittest
import numpy as np
from capability import archive,audit,failure,contribution_archive


class NativePrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads(archive.BASE.with_suffix('.json').read_text())
        cls.z=np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False)
        cls.sleep=next(r for r in cls.m['runs'] if r['name']=='quiet_b32_s100_r0')

    @classmethod
    def tearDownClass(cls):cls.z.close()

    def test_complete_archive_and_failed_prefix_recount(self):
        r=archive.audit_public();self.assertEqual(r['native_forwards'],7984);self.assertEqual(r['complete_learning_bank_snapshots'],7404)

    def test_failed128_stage_cannot_be_dropped(self):
        m=copy.deepcopy(self.m);m['runs']=[r for r in m['runs'] if r['name']!=archive.FAILED]
        with self.assertRaisesRegex(AssertionError,'including failed128'):archive.recount(m,self.z)

    def test_failed_stage_cannot_be_promoted_to_success(self):
        r=copy.deepcopy(next(r for r in self.m['runs'] if r['name']==archive.FAILED));z=self.data(r);r['status']='PASS'
        with self.assertRaises(AssertionError):failure.recount(r,z)

    def data(self,r):return archive.decode(r,{k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(r['name']+'__')})

    def test_one_driver_read_invalidates_sleeping_host_claim(self):
        z=self.data(self.sleep);z['host_activity']=z['host_activity'].copy();z['host_activity'][0,0]=1
        with self.assertRaisesRegex(AssertionError,'without driver polling'):audit.quiet(self.sleep,z)

    def test_changed_learning_output_is_rejected(self):
        z=self.data(self.sleep);z['scores']=z['scores'].copy();z['scores'][640,0]+=1
        with self.assertRaisesRegex(AssertionError,'quiet output'):audit.quiet(self.sleep,z)

    def test_partial_snapshot_deltas_cannot_fake_full_ram(self):
        r=self.sleep;p=r['name']+'__';z={k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(p)};z['bank0.delta.values']=z['bank0.delta.values'].copy();z['bank0.delta.values'][0]^=1
        with self.assertRaisesRegex(AssertionError,'snapshot checksum'):archive.decode(r,z)

    def test_forged_cpu_outputs_fail_independent_integer_reference(self):
        r=self.m['cpu_runs'][0];p=r['name']+'__';z={k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(p)};z['scores']=z['scores'].copy();z['scores'][0,0,0]+=1
        with self.assertRaisesRegex(AssertionError,'CPU exact primitive'):audit.cpu(r,z)

    def test_allocation_limit_cannot_be_claimed_as129(self):
        z={k:self.z[k] for k in self.z.files};z['allocation__requested_effective']=z['allocation__requested_effective'].copy();z['allocation__requested_effective'][-1,1]=129
        with self.assertRaises(AssertionError):archive.recount(self.m,z)

    def test_executed_source_version_is_retained_after_safety_repair(self):
        for p,info in self.m['executed_sources'].items():self.assertEqual(hashlib.sha256((archive.P/info['public_path']).read_bytes()).hexdigest(),info['sha256'],p)
        h=hashlib.sha256((archive.P/'src/quiet_relay.cpp').read_bytes()).hexdigest();self.assertEqual(h,self.m['post_close_safety']['published_source_sha256']);self.assertNotEqual(h,self.m['executed_sources']['src/quiet_relay.cpp']['sha256'])


class CNPContribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads(contribution_archive.BASE.with_suffix('.json').read_text());cls.z=np.load(contribution_archive.BASE.with_suffix('.npz'),allow_pickle=False)

    @classmethod
    def tearDownClass(cls):cls.z.close()

    def test_every_one_of11_distinct_regions_causally_checks(self):self.assertEqual(contribution_archive.audit_public()['distinct_causal_regions'],11)

    def test_no_causal_treatment_can_be_omitted(self):
        m=copy.deepcopy(self.m);m['runs'].pop()
        with self.assertRaisesRegex(AssertionError,'all11 causal treatments'):contribution_archive.recount(m,self.z)

    def test_wrong_physical_coordinate_is_rejected(self):
        m=copy.deepcopy(self.m);r=next(r for r in m['runs'] if r['name']=='suppress0');r['fixture_info']['packets'][1]['np']=[3,3,3]
        with self.assertRaisesRegex(AssertionError,'physical packet coordinates'):contribution_archive.recount(m,self.z)

    def test_changed_native_readback_fails(self):
        z={k:self.z[k] for k in self.z.files};key='suppress0__packet1.initial';z[key]=z[key].copy();z[key][2]^=1
        with self.assertRaisesRegex(AssertionError,'whole fixed memory'):contribution_archive.recount(self.m,z)

    def test_executed_sources_bound_to_measurement(self):
        for p,h in self.m['executed_sources'].items():self.assertEqual(binding_digest(archive.P,p),h,p)


class NativePresentation(unittest.TestCase):
    def test_video_and_presentation_sources_match_binding(self):
        m=json.loads((archive.P/'visual/NATIVE_PRIMITIVES_SOURCE_BINDING.json').read_text());self.assertEqual((m['fps'],m['seconds']),(60,24));self.assertIn('visual/native-primitives-4k60.mp4',m['hashes'])
        for p,h in m['hashes'].items():self.assertEqual(binding_digest(archive.P,p),h,p)

    def test_displayed_causal_counts_are_actual_native_values(self):
        r=json.loads((archive.P/'visual/native-primitives-replay.json').read_text())
        with np.load(contribution_archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:
            i=int(np.flatnonzero(z['baseline__inputs'].sum(1)==512)[0]);self.assertEqual(r['baseline_output'],z['baseline__scores'][i].tolist());self.assertEqual(len(r['causal_cases']),11)
            for case in r['causal_cases']:self.assertEqual(case['measured_output'],z[case['name']+'__scores'][i].tolist());self.assertEqual(case['measured_drop'],(z['baseline__scores'][i]-z[case['name']+'__scores'][i]).tolist())

    def test_connection_spokes_match_captured_ram_boundaries(self):
        replay=json.loads((archive.P/'visual/native-primitives-replay.json').read_text());m=json.loads(archive.BASE.with_suffix('.json').read_text());r=next(r for r in m['runs'] if r['name']=='quiet_b32_s100_r0')
        with np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:a=archive.decode(r,{k.split('__',1)[1]:z[k] for k in z.files if k.startswith(r['name']+'__')})
        self.assertEqual(len(replay['quiet_recognition_records']),8)
        for n in replay['quiet_recognition_records']:
            for index,conn in zip(replay['quiet_capture_indices'],n['connections_at_boundaries']):
                at=int(np.flatnonzero(a['boundary_indices']==index)[0]);start=2+n['record']*20+4;w=a['bank0.snapshots'][at,start:start+16];want=[j for j in range(512) if int(w[j//32])>>(j%32)&1];self.assertEqual(conn,want)


if __name__=='__main__':unittest.main()
