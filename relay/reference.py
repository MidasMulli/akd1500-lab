"""Independent exact binary-learning reference plus explicit native message contract."""
import numpy as np
from pong.relearn.model import Reference
from relay.design import N, U, NW, CLASSES, THRESHOLD, initial, schedule, metrics, SEEDS, ARMS, gate


class Network:
    def __init__(self, ram):
        self.banks = [Reference(r, N, U, NW, CLASSES) for r in ram]

    def forward(self, x, label, mask):
        p = self.banks[0].forward(x, label) if mask & 1 else self.scores(0, x)
        activity = np.asarray(p) > THRESHOLD
        support = activity.reshape(CLASSES, -1).sum(axis=1)
        identity = int(support.argmax())
        message = np.zeros(N, np.uint8); message[identity * 128:(identity + 1) * 128] = 1
        p2 = self.banks[1].forward(message, label) if mask & 2 else self.scores(1, message)
        output = (np.asarray(p2) > THRESHOLD).reshape(CLASSES, -1).sum(axis=1)
        return output.tolist(), dict(identity=identity, recognition_scores=p, routing_scores=p2,
                                     recognition_support=support.tolist())

    def scores(self, bank, x):
        bits = sum(1 << int(k) for k in np.flatnonzero(x))
        return [(w & bits).bit_count() for w in self.banks[bank].weights]


def replay(seed, arm, ram=None, snapshots=False):
    x, rows = schedule(arm); net = Network(initial(seed) if ram is None else ram)
    outputs, details, states = [], [], []
    downstream_start = None
    for values, row in zip(x, rows):
        if row['phase'] == 'repair' and downstream_start is None:
            downstream_start = net.banks[1].ram[:]
        y, detail = net.forward(values, row['label'], row['learning_mask'])
        outputs.append(y); details.append(detail)
        if snapshots: states.append([b.ram[:] for b in net.banks])
    report = dict(seed=seed, arm=arm, metrics=metrics(rows, outputs),
                  downstream_unchanged_during_repair=downstream_start == net.banks[1].ram)
    return report, outputs, details, states


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(); p.add_argument('output', type=Path); a = p.parse_args()
    results = [replay(seed, arm)[0] for seed in SEEDS for arm in ARMS]
    gates = [gate(results[i], results[i+1]) for i in range(0, len(results), 3)]
    z = dict(status='CPU_FORECAST_ONLY', results=results, gates=gates)
    a.output.write_text(json.dumps(z, indent=2) + '\n'); print(json.dumps(z), flush=True)
