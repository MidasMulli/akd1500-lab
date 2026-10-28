import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from instruments.explorer.server import make_handler, HERE


class FakeBridge:
    enable_native=False
    def __init__(self):self.calls=[]
    def snapshot(self):return dict(status='idle',events=[])
    def command(self,a):self.calls.append(a);return dict(status='test')


class HttpBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge=FakeBridge()
        # Bind before constructing the handler so its allowed Host is exact.
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),http.server.BaseHTTPRequestHandler)
        cls.port=cls.server.server_port
        cls.server.RequestHandlerClass=make_handler(cls.bridge,HERE/'generated/replay.json',cls.port,'192.0.2.1')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def request(self,method,path,body=None,headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        c.request(method,path,body=body,headers=headers or {});r=c.getresponse();code=r.status;raw=r.read();c.close();return code,raw

    def test_replay_and_status_never_touch_hardware(self):
        count=len(self.bridge.calls)
        self.assertEqual(self.request('GET','/api/replay')[0],200)
        self.assertEqual(self.request('GET','/api/status')[0],200)
        self.assertEqual(len(self.bridge.calls),count)

    def test_tablet_versions_are_read_only_and_controls_stay_disabled(self):
        count=len(self.bridge.calls)
        for path,schema in [('/api/tab5',1),('/api/tab5/v2',2)]:
            code,raw=self.request('GET',path+'?event=141&bank=0&record=39')
            self.assertEqual(code,200)
            body=json.loads(raw);self.assertEqual(body['schema'],schema)
            self.assertFalse(body['controls_enabled']);self.assertEqual(len(body['banks']),3)
        self.assertEqual(len(self.bridge.calls),count)

    def test_cross_site_and_missing_token_rejected(self):
        code,raw=self.request('GET','/api/status');token=json.loads(raw)['csrf']
        request=json.dumps(dict(action='query',key=0))
        self.assertEqual(self.request('POST','/api/live',request,{'Content-Type':'application/json'})[0],403)
        self.assertEqual(self.request('POST','/api/live',request,{'X-Explorer-Token':token,'Origin':'https://foreign.invalid'})[0],403)

    def test_unknown_host_and_paths_rejected(self):
        self.assertEqual(self.request('GET','/api/status',headers={'Host':'foreign.invalid'})[0],421)
        self.assertEqual(self.request('GET','/../live.py')[0],404)
        self.assertEqual(self.request('GET','/.superdesign/design-system.md')[0],404)

    def test_export_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.json';p.write_text('{}')
            p.with_suffix('.receipt.json').write_text(json.dumps({'data_sha256':'bad'}))
            with self.assertRaises(ValueError):make_handler(FakeBridge(),p,8892,'192.0.2.1')


if __name__=='__main__':unittest.main()
