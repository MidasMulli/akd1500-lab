"""Serial confirmation of the prospectively selected host-service setting."""
import argparse,json
from coprocessor.service_run import run,cell,good,S,sha

def execute(mode,seed,arm,repeat=0,sleep=0,count=64):
    name=cell(mode,64,seed,'potentials',arm,count,repeat,sleep,1,1,1)
    if good(name):return
    run(argparse.Namespace(mode=mode,side=64,seed=seed,contract='potentials',arm=arm,count=count,repeat=repeat,sleep=sleep,batch=1,profile=1,full=1))

def main():
    selected=json.loads((S/'SELECTION.json').read_text())
    assert selected['candidate']==dict(sleep_us=100,batch=1,profile=1,contract='potentials')
    for seed in [107601,107701]:
        execute('native',seed,0,count=8)
        for sleep in [0,100]:
            for arm in [1,0]:execute('native',seed,arm,sleep=sleep)
    for repeat in [1,2]:
        for j,seed in enumerate([107501,107601,107701]):
            for sleep in ([100,0] if (j+repeat)%2 else [0,100]):
                for arm in ([0,1] if (j+repeat)%2 else [1,0]):execute('native',seed,arm,repeat,sleep)
            modes=selected['confirmation']['cpu_controls'];offset=(j+repeat)%len(modes)
            for mode in modes[offset:]+modes[:offset]:
                for arm in ([0,1] if (j+repeat)%2 else [1,0]):execute(mode,seed,arm,repeat)
    print(json.dumps(dict(status='CONFIRMATION_COMPLETE',selection_sha256=sha(S/'SELECTION.json'))),flush=True)

if __name__=='__main__':main()
