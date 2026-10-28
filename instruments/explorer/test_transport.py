"""Public transport configuration cannot become SSH options or shell syntax."""
import os,shlex,unittest
from unittest.mock import patch
from instruments.explorer.server import Bridge

class Transport(unittest.TestCase):
    def test_no_private_default_or_option_host(self):
        for host in ['', '-oProxyCommand=bad', 'host;touch /tmp/no', 'host\nother']:
            b=Bridge(True)
            with patch.dict(os.environ,{'AKD_SSH_HOST':host,'AKD_REMOTE_ROOT':'/lab','AKD_REMOTE_PYTHON':'/bin/python'},clear=True),patch('subprocess.Popen')as popen:
                with self.assertRaisesRegex(ValueError,'Configure AKD_SSH_HOST'):b.command(dict(action='start',mode='native',arm='native'))
                popen.assert_not_called();self.assertEqual(b.snapshot()['status'],'idle')

    def test_remote_paths_are_shell_quoted(self):
        class Dummy:
            stdin=type('Input',(),{'write':lambda *a:b.state.update(status='ready'),'flush':lambda *a:None})()
            def poll(self):return None
        b=Bridge(True)
        def fake_launch(argv,**kw):
            self.assertEqual(argv[-2],'user@nano.example');words=shlex.split(argv[-1]);self.assertEqual(words[1],'/lab folder;literal');self.assertEqual(words[4],'/python folder/bin/python');b.state['status']='ready';return Dummy()
        def no_thread(*a,**kw):
            return type('Thread',(),{'start':lambda *a:None})()
        env={'AKD_SSH_HOST':'user@nano.example','AKD_REMOTE_ROOT':'/lab folder;literal','AKD_REMOTE_PYTHON':'/python folder/bin/python'}
        with patch.dict(os.environ,env,clear=True),patch('subprocess.Popen',side_effect=fake_launch),patch('threading.Thread',side_effect=no_thread):b.command(dict(action='start',mode='native',arm='native'))

if __name__=='__main__':unittest.main()
