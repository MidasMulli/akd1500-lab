"""Append-only new fixture suffixes; never overwrite the qualified main fixtures."""
from discovery.run import *
def main():
    for seed in SEEDS:
        f=F/f's{seed}';ram=np.fromfile(f/'initial.ram','<u4')
        for arm in (['unseen','unseen_frozen'] if seed==SEEDS[0] else ['unseen']):
            a=simulate(ram,12,arm);stem=f/f'{arm}_c12'
            assert not stem.with_suffix('.plan').exists()
            np.asarray([[e['bit'],e['label'],e['sever']] for e in a['events']],'<i4').tofile(stem.with_suffix('.plan'))
            for k in ['inputs','features','outputs','potentials','states','cpu']:a[k].tofile(stem.with_suffix('.'+k))
            print(seed,arm,len(a['events']),assess(a)['unique_records'])
if __name__=='__main__':main()
