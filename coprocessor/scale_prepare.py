"""Extend the qualified arithmetic without changing its precision or topology."""
import argparse, json
from coprocessor.scale_run import S,F,WIDTHS,SEEDS,passed

def main(c,seed):
    assert c in WIDTHS and seed in SEEDS
    assert passed(S/'reference_raw/AllNps_n64')
    state=json.loads((S/'STATE.json').read_text());assert not state['closed'] and not state.get('failure')
    if c!=512:
        prev=WIDTHS[WIDTHS.index(c)-1]
        for arm in [0,1]:assert passed(S/'raw'/f'native_c{prev}_s107501_a{arm}_n64_r0')
    from coprocessor.resident_prepare import main as prepare
    prepare(F/f'c{c}_s{seed}',c,seed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('filters',type=int,choices=WIDTHS);p.add_argument('seed',type=int,choices=SEEDS)
    a=p.parse_args();main(a.filters,a.seed)
