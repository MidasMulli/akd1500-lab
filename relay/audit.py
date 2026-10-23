"""Recount actual per-event RAM and final outputs; do not infer success from exit code."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from relay.design import schedule, metrics
from relay.reference import Network

P = Path(__file__).resolve().parents[1]


def words(path): return np.frombuffer(path.read_bytes(), '<u4').astype(np.uint64).tolist()


def verify(run):
    rec = json.loads((run / 'RECEIPT.json').read_text())
    assert rec['status'] == 'PASS' and rec['device_released']
    for name, expected in rec['hashes'].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected, ('raw changed', name)
    full_x, full_rows = schedule(rec['arm']); rows = json.loads((run / 'stream.rows.json').read_text())
    ii = [r['original_index'] for r in rows]
    x = np.frombuffer((run / 'stream.inputs').read_bytes(), np.uint8).reshape(-1, 512)
    labels = np.frombuffer((run / 'stream.labels').read_bytes(), '<i4')
    masks = np.frombuffer((run / 'stream.masks').read_bytes(), np.uint8)
    assert np.array_equal(x, full_x[ii]) and len(x) == rec['samples']
    for i, row in enumerate(rows):
        assert {k: v for k, v in row.items() if k != 'original_index'} == full_rows[ii[i]]
        assert labels[i] == row['label'] and masks[i] == row['learning_mask']
    f = P / rec['fixture']; ram = [words(f.parent / (f.name + f'.bank{k}.ram')) for k in range(2)]
    net = Network(ram)
    for k in range(2): assert words(run / f'native.bank{k}.initial.ram') == ram[k]
    native = np.frombuffer((run / 'native.scores').read_bytes(), '<i4').reshape(-1, 4)
    assert len(native) == len(rows)
    detail, edits, snapshots = [], [0, 0], [[], []]; before_repair = None
    for i, row in enumerate(rows):
        if row['phase'] == 'repair' and before_repair is None: before_repair = [b.ram[:] for b in net.banks]
        before = [b.ram[:] for b in net.banks]
        y, response = net.forward(x[i], int(labels[i]), int(masks[i]))
        assert y == native[i].tolist(), ('final output mismatch', i)
        for k, bank in enumerate(net.banks):
            actual = words(run / f'native.snapshot_{i}_bank{k}.ram')
            assert actual == bank.ram, ('independent native RAM mismatch', i, k)
            if not masks[i] & (1 << k): assert actual == before[k], ('frozen bank changed', i, k)
            edits[k] += sum((a ^ b).bit_count() for j, (a, b) in enumerate(zip(before[k], actual))
                            if j >= 2 and (j - 2) % 20 >= 4)
            snapshots[k].append(actual)
        detail.append(response)
    for k in range(2): assert words(run / f'native.bank{k}.ram') == net.banks[k].ram
    for j in range(3):
        fixed = words(f.parent / (f.name + f'.fixed{j}.ram'))
        assert words(run / f'native.fixed{j}.initial.ram') == words(run / f'native.fixed{j}.final.ram') == fixed
    repair_unchanged = before_repair is not None and before_repair[1] == net.banks[1].ram
    report = dict(status='PASS', run=run.name, seed=rec['seed'], arm=rec['arm'], smoke=rec['smoke'],
                  samples=len(rows), exact_native_output_values=int(native.size), measured_full_bank_snapshots=2 * len(rows),
                  connection_bit_edits=edits, downstream_unchanged_during_repair=repair_unchanged,
                  recognition_unchanged_during_repair=before_repair is not None and before_repair[0] == net.banks[0].ram,
                  metrics=metrics(rows, native) if not rec['smoke'] else {}, fixed_communication_unchanged=True,
                  intermediate_responses='independently reconstructed, not electrical measurements',
                  verified_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run / 'VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); print(json.dumps(verify(p.parse_args().run)), flush=True)
