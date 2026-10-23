"""Check the published live frame/feature/packet binding without a ROM or device."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .bank import encode

ROOT = Path(__file__).resolve().parents[2]


def audit(root=ROOT, replay=None):
    z = json.loads((root / 'visual/mario-live-data.json').read_text()) if replay is None else replay
    archive = root / 'results/2026-10-23-mario-sealed.npz'
    game = root / 'visual/mario-live-game.npz'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(archive) == z['raw_archive_sha256']
    assert sha(game) == z['game_sha256']
    assert 0 <= z['alignment_uncertainty_ns'] < 1000000
    assert z['origin_upper_ns'] - z['origin_lower_ns'] == z['alignment_uncertainty_ns']
    source = np.load(game, allow_pickle=False)
    frames, clock = source['frames'], source['mono_offsets_ns']
    native = np.load(archive, allow_pickle=False)
    inputs = native['fixture__live_full01__inputs'].reshape(-1, 1024)
    labels = native['fixture__live_full01__labels']
    scores = native['live_full01__native_scores'].reshape(-1, 192)
    events = z['events']
    rows = {e['frame_id']: e for e in events if e['kind'] == 'frame'}
    assert set(rows) == set(range(len(frames)))
    by_elapsed = {}
    for i, row in rows.items():
        by_elapsed.setdefault((row['case'], row['elapsed']), i)
        assert row['mono_ns'] >= z['origin_lower_ns'] + int(clock[i])
    checked = 0
    for event in events:
        if event['kind'] != 'response_pair':
            continue
        frame_id = event['input_frame']; row = rows[frame_id]
        assert event['elapsed'] == row['elapsed'], 'response source-frame clock mismatch'
        previous = frames[by_elapsed[row['case'], max(0, row['elapsed'] - 8)]]
        sky = frames[by_elapsed[row['case'], 0]][50, 0].astype(np.int16)
        for action, reply in enumerate(event['replies']):
            i = reply['index']; expected = encode(frames[frame_id], previous, sky, action, 'projection', row['elapsed'])
            assert np.array_equal(expected, inputs[i]), 'response source-frame features mismatch'
            assert hashlib.sha256(expected.tobytes()).hexdigest() == reply['input_sha256']
            assert labels[i] == reply['label'] == 2
            assert scores[i].tolist() == reply['scores'], 'published response differs from native raw scores'
            assert reply['received_monotonic_ns'] >= reply['sent_monotonic_ns'] >= z['origin_lower_ns'] + int(clock[frame_id])
            checked += 1
    return dict(status='PASS', reconstructed_response_inputs=checked, frames=len(frames),
                alignment_uncertainty_ns=z['alignment_uncertainty_ns'])


if __name__ == '__main__':
    print(json.dumps(audit()))
