"""Read-only tablet projections checked against retained native numeric arrays."""
import copy
import json
from pathlib import Path
from urllib.parse import parse_qs
import unittest
import numpy as np
from instruments.explorer.tablet import compact

ROOT=Path(__file__).resolve().parents[2]


class TabletProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy=json.loads((ROOT/'instruments/explorer/generated/replay.json').read_text())
        cls.four=json.loads((ROOT/'instruments/explorer/generated/20261026-pending-eight.json').read_text())
        cls.meta=json.loads((ROOT/'results/20261026-pending-eight.json').read_text())
        cls.archive=np.load(ROOT/'results/20261026-pending-eight.npz',allow_pickle=False)
        cls.native_run=next(r for r in cls.meta['runs'].values()
                     if r['receipt']['mode']=='native' and r['receipt']['seed']==186101
                     and r['receipt']['arm']=='native' and r['receipt']['stage']=='pilot')

    @classmethod
    def tearDownClass(cls):cls.archive.close()

    def test_legacy_shape_remains_v1(self):
        p=compact(self.legacy,{},parse_qs('event=141&bank=0&record=39'))
        self.assertEqual(p['schema'],1);self.assertEqual(len(p['banks']),3)
        self.assertEqual(len(p['selection']['after']['connections']),32)
        self.assertNotIn('trace_flags',p['event']);self.assertNotIn('classes',p['banks'][0])
        with self.assertRaisesRegex(ValueError,'v2'):compact(self.four,{},parse_qs('bank=0'))

    def test_four_bank_selection_matches_numeric_capture(self):
        # Include all trace class partitions, a changed policy record, and helper.
        for event,bank,record in [(192,3,24),(193,0,2),(200,3,120),(202,3,56),(208,3,88),(96,3,16)]:
            p=compact(self.four,{},parse_qs(f'event={event}&bank={bank}&record={record}'),schema=2)
            self.assertEqual(p['schema'],2);self.assertEqual(p['study'],'20261026-pending-eight')
            b=p['banks'][bank];sel=p['selection'];width=b['inputs'];stride=4+width//32
            raw=self.archive[self.native_run['arrays'][f'states{bank}']['key']].reshape(536,-1)
            for field,i in [('before',event-1),('after',event)]:
                start=2+record*stride;native=raw[i,start:start+stride]
                self.assertEqual(sel[field]['words'],native[:4].tolist())
                bits=np.unpackbits(native[4:].copy().view('u1'),bitorder='little')
                self.assertEqual(sel[field]['connections'],np.flatnonzero(bits).tolist())
                self.assertEqual(len(sel[field]['connections']),b['num_weights'])
            self.assertEqual(sel['class_id'],record//(b['units']//b['classes']))
            self.assertEqual(set(sel['added']),set(sel['after']['connections'])-set(sel['before']['connections']))
            self.assertEqual(set(sel['removed']),set(sel['before']['connections'])-set(sel['after']['connections']))
            self.assertLess(len(json.dumps(p).encode()),16384)
            self.assertFalse(p['controls_enabled'])

    def test_trace_flags_keep_measured_class_partition(self):
        for event in [192,200,208,216]:
            p=compact(self.four,{},parse_qs(f'event={event}&bank=3&record=24'),schema=2)
            recorded=self.archive[self.native_run['arrays']['scores']['key']].reshape(536,128)[event]
            flags=[int(recorded[32*c+24:32*c+32].any()) for c in range(4)]
            self.assertEqual(p['event']['trace_flags'],flags)
            per_record=[int(any(r['output'] for r in p['banks'][3]['records'] if r['class_id']==c)) for c in range(4)]
            self.assertEqual(per_record,flags)

    def test_four_bank_live_fails_closed_even_with_a_three_bank_stream(self):
        p=compact(self.four,self.legacy['arms']['native'],parse_qs('mode=live'),schema=2)
        self.assertFalse(p['available']);self.assertFalse(p['controls_enabled'])
        self.assertEqual(p['status'],'unsupported_live_graph');self.assertEqual(p['banks'],[])

    def test_selector_and_dimension_boundaries(self):
        for raw in ['bank=4','bank=3&record=128','bank=-1','record=true','schema=1','bank=0&bank=3','arm=open']:
            with self.assertRaises(ValueError):compact(self.four,{},parse_qs(raw),schema=2)
        for field,value in [('inputs',1024),('num_weights',64),('classes',3)]:
            d=copy.deepcopy(self.four);d['banks'][3][field]=value
            with self.assertRaises(ValueError):compact(d,{},parse_qs('bank=3'),schema=2)


if __name__=='__main__':unittest.main()
