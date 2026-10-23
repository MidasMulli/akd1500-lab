"""Loss-detecting raw serial capture. Does not flash or reset the recorder."""
from pathlib import Path
import argparse,json,time
def collect(port,output,seconds):
    import serial
    output.mkdir(parents=True,exist_ok=False)
    # Set control lines before opening to avoid common ESP automatic reset paths.
    ser=serial.Serial();ser.port=port;ser.baudrate=115200;ser.timeout=.2;ser.dtr=False;ser.rts=False
    ser.open();start=time.monotonic();last=None;header=False
    try:
        with (output/'serial.jsonl').open('w') as f,(output/'serial.raw').open('wb') as raw:
            while time.monotonic()-start<seconds:
                line=ser.readline();stamp=time.monotonic_ns()
                if not line:continue
                raw.write(line);raw.flush()
                try:row=json.loads(line)
                except (ValueError,UnicodeError):raise RuntimeError('Malformed recorder output; preserved raw bytes')
                if row['kind']=='recorder':
                    if header:raise RuntimeError('Recorder restarted during capture')
                    header=True
                elif row['kind']=='sample':
                    if not header:raise RuntimeError('Missing recorder identity header; start a fresh explicit recorder stream')
                    if last is not None and row['sequence']!=last+1:raise RuntimeError('Missing or duplicate power sample')
                    last=row['sequence']
                elif row['kind']=='recorder_error':raise RuntimeError(row['error'])
                f.write(json.dumps(dict(host_receive_monotonic_ns=stamp,record=row))+'\n');f.flush()
    finally:ser.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);a=p.parse_args()
    assert 0<a.seconds<=3600;collect(a.port,a.output,a.seconds)
