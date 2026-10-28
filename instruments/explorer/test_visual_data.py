"""RGB profile source binding and rejection of accidental live/tablet actions."""
import base64,hashlib,io,json,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from PIL import Image
from instruments.explorer.server import Bridge,make_handler
ROOT=Path(__file__).resolve().parents[2];FILE=ROOT/'instruments/explorer/generated/20261026-visual-memory-v4.json';S=ROOT/'studies/20261026-visual-memory-v4'

def apply_record(ram,j,rec):
    a=ram[2:].reshape(220,36);a[j]=0;a[j,:4]=rec['words']
    for bit in rec['connections']:a[j,4+bit//32]|=np.uint32(1<<(bit%32))

@unittest.skipUnless((S/'raw/native-fresh-adaptive').exists(), 'Optional full RGB captures are not in the replay checkout')
class VisualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=json.loads(FILE.read_text())
    def test_every_record_delta_matches_actual_full_RAM(self):
        for tag in ['adaptive','frozen']:
            f=S/'native-fixtures'/('fresh-'+tag);raw=S/'raw'/('native-fresh-'+tag);initial=np.fromfile(f/'cell.ram','<u4');state=initial.copy()
            for rec in self.data['arms'][tag]['initial'][0]:apply_record(state,rec['id'],rec)
            np.testing.assert_array_equal(state,initial);want=np.fromfile(raw/'result.snapshots','<u4').reshape(520,-1)
            for i,e in enumerate(self.data['arms'][tag]['events']):
                if not e['teacher_enabled']:self.assertEqual(e['changes'],[])
                for c in e['changes']:
                    j=c['record'];old=state.copy();apply_record(old,j,c['before']);np.testing.assert_array_equal(old,state);apply_record(state,j,c['after'])
                    self.assertEqual(len(c['after']['connections']),64)
                np.testing.assert_array_equal(state,want[i])
    def test_captured_scores_and_host_argmax(self):
        for tag in ['adaptive','frozen']:
            actual=np.fromfile(S/'raw'/('native-fresh-'+tag)/'result.scores','<i4').reshape(520,220)
            events=self.data['arms'][tag]['events'];np.testing.assert_array_equal([e['observed_potentials'][0] for e in events],actual)
            np.testing.assert_array_equal([e['prediction'] for e in events],actual[:,:200].reshape(520,10,20).max(2).argmax(1))
            for e in events:
                self.assertIsNone(e['policy_flags']);self.assertIsNone(e['memory_flags'])
                if not e['teacher_enabled']:self.assertIsNone(e['teaching_label'])
    def test_images_are_exact_recorded_pixels(self):
        self.assertEqual(len(self.data['images']),400)
        raw=np.fromfile(S/'native-fixtures/fresh-adaptive/cell.inputs','u1').reshape(520,128,128,3)
        for i,e in enumerate(self.data['arms']['adaptive']['events']):
            item=self.data['images'][e['image_id']];im=np.array(Image.open(io.BytesIO(base64.b64decode(item['data_url'].split(',',1)[1]))));np.testing.assert_array_equal(im,raw[i]);self.assertEqual(hashlib.sha256(im.tobytes()).hexdigest(),item['raw_rgb_sha256'])
    def test_matched_prefix_and_arm_labels(self):
        a,b=[self.data['arms'][k] for k in ['adaptive','frozen']];self.assertEqual(a['initial'],b['initial']);self.assertEqual([e['changes'] for e in a['events'][:80]],[e['changes'] for e in b['events'][:80]])
        self.assertTrue(all(not e['teacher_enabled'] for e in b['events'][80:]));self.assertFalse(self.data['live_compatible']);self.assertEqual(self.data['schema'],3)

class VisualHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge=Bridge(False);cls.calls=[]
        def forbidden(a):cls.calls.append(a);raise AssertionError('RGB replay reached hardware bridge')
        cls.bridge.command=forbidden;cls.http=ThreadingHTTPServer(('127.0.0.1',0),lambda *a:None);cls.port=cls.http.server_address[1];cls.http.RequestHandlerClass=make_handler(cls.bridge,FILE,cls.port,'192.0.2.1');cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start();cls.base=f'http://127.0.0.1:{cls.port}'
    @classmethod
    def tearDownClass(cls):cls.http.shutdown();cls.http.server_close();cls.thread.join()
    def test_new_profile_refuses_native_enable(self):
        with self.assertRaisesRegex(ValueError,'no qualified native actor'):make_handler(Bridge(True),FILE,self.port,'192.0.2.1')
    def test_four_bank_replay_cannot_enable_three_bank_actor(self):
        four=ROOT/'instruments/explorer/generated/20261026-pending-eight.json'
        with self.assertRaisesRegex(ValueError,'no qualified native actor'):make_handler(Bridge(True),four,self.port,'192.0.2.1')
    def test_status_disabled_and_live_action_never_reaches_bridge(self):
        status=json.load(urlopen(self.base+'/api/status'));self.assertFalse(status['native_enabled']);request=Request(self.base+'/api/live',data=b'{"action":"start","arm":"native","mode":"native"}',headers={'Content-Type':'application/json','X-Explorer-Token':status['csrf'],'Origin':self.base})
        with self.assertRaises(HTTPError) as err:urlopen(request)
        self.assertEqual(err.exception.code,409);self.assertEqual(self.calls,[])
    def test_no_wrong_tablet_or_live_projection(self):
        for path,code in [('/api/tab5',400),('/api/tab5/v2',400),('/api/live',409)]:
            with self.assertRaises(HTTPError) as err:urlopen(self.base+path)
            self.assertEqual(err.exception.code,code)

if __name__=='__main__':unittest.main()
