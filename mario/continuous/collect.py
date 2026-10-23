"""Prospectively fixed short-horizon experience, including invalid prefixes."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from mario.sealed.bank import encode, save, sha
from mario.continuous.environment import contexts, opened, observation, button


def collect(output):
    output.mkdir(parents=True, exist_ok=False); rows = []; arrays = {}
    for c in contexts('train'):
        for a in [0, 1]:
            env, history, sky, info, valid = opened(c)
            try:
                row = dict(**c, action=a, valid=valid)
                if valid:
                    x = encode(**observation(history, sky), action=a, encoder='projection', elapsed=c['wait']+c['start'])
                    arrays[f'{c["id"]}_{a}'] = x; start = int(info['x_pos']); dead = False
                    for k in range(24):
                        _, _, terminated, truncated, info = env.step(button(a, k))
                        if info.get('death') or terminated or truncated: dead = True; break
                    row.update(success=bool(not dead and int(info['x_pos'])-start >= 12), advance_evaluation_only=int(info['x_pos'])-start,
                               input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
                rows.append(row)
            finally: env.close()
    np.savez_compressed(output/'inputs.npz', **arrays)
    save(output/'EXPERIENCE.json', dict(status='COLLECTED', rows=rows, acting_game_RAM=False, horizon=24, source_sha256=sha(__file__)))
    print(json.dumps(dict(contexts=len(rows)//2, valid=sum(r['valid'] for r in rows)//2, labels=[sum(r.get('success') == v for r in rows if r['valid']) for v in [False, True]])))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();collect(a.output.resolve())
