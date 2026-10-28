from media_binding import binding_digest
"""Native repair, retained maps, complete RAM deltas and preserved source identity."""
import copy
import hashlib
import json
import unittest
import numpy as np
from resident import or_archive,or_run,or_recruitment


class ORRelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(or_archive.BASE.with_suffix('.json').read_text())
        cls.z=np.load(or_archive.BASE.with_suffix('.npz'),allow_pickle=False)
        cls.rec=next(r for r in cls.meta['runs'] if r['name']=='full_u512_s102101_repair')

    @classmethod
    def tearDownClass(cls):cls.z.close()

    def test_every_complete_native_state_and_output_recounts(self):
        r=or_archive.audit_public();self.assertEqual(r['full_bank_snapshots'],20016);self.assertEqual(r['native_forwards'],6672)

    def test_missing_confirmation_arm_is_rejected(self):
        m=copy.deepcopy(self.meta);m['runs'].pop()
        with self.assertRaisesRegex(AssertionError,'complete OR-relay'):or_archive.recount(m,self.z)

    def test_connection_delta_edit_fails_original_snapshot_checksum(self):
        r=self.rec;prefix=r['name']+'__';z={k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(prefix)}
        z['bank0.delta.values']=z['bank0.delta.values'].copy();z['bank0.delta.values'][0]^=1
        with self.assertRaisesRegex(AssertionError,'complete snapshot checksum'):or_archive.expand_run(r,z)

    def test_dropping_delta_boundary_is_rejected(self):
        r=self.rec;prefix=r['name']+'__';z={k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(prefix)}
        z['bank1.delta.offsets']=z['bank1.delta.offsets'][:-1]
        with self.assertRaises(AssertionError):or_archive.expand_run(r,z)

    def test_changed_native_query_output_is_rejected(self):
        r=self.rec;prefix=r['name']+'__';z=or_archive.expand_run(r,{k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(prefix)})
        z['native.scores']=z['native.scores'].copy();z['native.scores'][640,0]+=1
        with self.assertRaisesRegex(AssertionError,'output'):or_run.recount(r,z)

    def test_repairs_and_retention_across_three_initializations(self):
        rr=[r for r in self.meta['runs'] if not r['smoke'] and r['arm']=='repair']
        self.assertEqual(len(rr),3)
        for r in rr:
            self.assertEqual(r['audit']['metrics']['changed_after']['correct'],64)
            self.assertEqual(r['audit']['metrics']['retention']['correct'],64)
            self.assertTrue(r['audit']['protected_banks_unchanged'])
            self.assertEqual(r['connection_stats']['repair_connection_edit_events'][1:],[0,0])

    def test_allocated_records_are_not_all_edited_records(self):
        s=self.rec['connection_stats'];self.assertEqual(s['allocated_learning_records'],1536)
        self.assertEqual([len(a) for a in s['unique_records_with_connection_edits']],[8,4,4])
        wrong=next(r for r in self.meta['runs'] if r['arm']=='wrong_bank')
        self.assertEqual(wrong['connection_stats']['repair_connection_edit_events'],[0,0,0])

    def test_native_sources_match_executed_freeze(self):
        for p,h in self.meta['executed_sources'].items():self.assertEqual(binding_digest(or_archive.P,p),h,p)

    def test_new_native_records_preserve_all_old_record_words(self):
        actual=or_recruitment.derive(self.meta,self.z)
        self.assertEqual(actual,json.loads(or_recruitment.OUT.read_text()))
        for seed in actual['seeds']:
            self.assertTrue(seed['all_old_record_words_unchanged_after_initial_teaching'])
            self.assertTrue(set(seed['old_connection_edited_records']).isdisjoint(seed['new_connection_edited_records']))

    def test_film_sources_are_bound_to_numeric_evidence(self):
        binding=json.loads((or_archive.P/'visual/FNP_OR_RELAY_SOURCE_BINDING.json').read_text())
        self.assertEqual((binding['duration_seconds'],binding['display_fps']),(24,60))
        self.assertIn('visual/fnp-or-relay-4k60.mp4',binding['hashes'])
        for p,h in binding['hashes'].items():
            self.assertEqual(binding_digest(or_archive.P,p),h,p)

    def test_replay_events_use_measured_outputs_and_current_record_changes(self):
        replay=json.loads((or_archive.P/'visual/fnp-or-relay-replay.json').read_text())
        r=self.rec;prefix=r['name']+'__';raw=or_archive.expand_run(r,{k.split('__',1)[1]:self.z[k] for k in self.z.files if k.startswith(prefix)})
        self.assertEqual([e['index'] for e in replay['events']],list(range(384,768)))
        for e in replay['events']:
            i=e['index'];self.assertEqual(e['measured_output'],raw['native.scores'][i].tolist())
            for k,b in enumerate(e['banks']):
                before=raw[f'bank{k}.snapshots'][i-1,2:].reshape(512,20)[:,4:]
                after=raw[f'bank{k}.snapshots'][i,2:].reshape(512,20)[:,4:]
                ids=np.flatnonzero((before!=after).any(1)).tolist()
                self.assertEqual(b['edited_records'],ids)


if __name__=='__main__':unittest.main()
