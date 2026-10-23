"""Independent replay of native RAM, rendered game frames, features and action causality."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from bank import encode, sha, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from verify import verify, read_words


def audit(study, capture, raw):
    receipt = json.loads((capture / 'LIVE_RECEIPT.json').read_text())
    assert receipt['status'] == 'LIVE_PHYSICAL_CAPTURE_PASS'
    assert sha(capture / 'game.npz') == receipt['game_sha256']
    assert sha(capture / 'events.jsonl') == receipt['events_sha256']
    native = verify(ROOT, raw)
    saved = json.loads((raw / 'RECEIPT.json').read_text())
    n = int((raw / 'fixture.meta').read_text().split()[0])
    inputs = np.frombuffer((raw / 'fixture.inputs').read_bytes(), np.uint8).reshape(-1, n)
    labels = np.frombuffer((raw / 'fixture.labels').read_bytes(), '<i4')
    scores = np.frombuffer((raw / 'native.scores').read_bytes(), '<i4').reshape(-1, 192)
    packets = {r['id']: r for r in map(json.loads, (raw / 'native.stdout').read_text().splitlines()) if r['kind'] == 'forward'}
    events = [json.loads(line) for line in (capture / 'events.jsonl').read_text().splitlines()]
    z = np.load(capture / 'game.npz', allow_pickle=False)
    frames, clock = z['frames'], z['mono_offsets_ns']
    assert len(frames) == len(clock) == receipt['frames'] and np.all(np.diff(clock) > 0)
    frame_events = {e['frame_id']: e for e in events if e['kind'] == 'frame'}
    assert set(frame_events) == set(range(len(frames)))
    by_elapsed = {}
    for i, event in frame_events.items():
        by_elapsed.setdefault((event['case'], event['elapsed']), i)
    initial = read_words(raw / 'fixture.ram')
    checked, queries = set(), 0
    def check(reply):
        nonlocal queries
        measured = packets[reply['id']]
        assert all(reply[k] == v for k, v in measured.items()), 'live packet differs from raw device reply'
        i = reply['index']
        assert reply['input_sha256'] == hashlib.sha256(inputs[i].tobytes()).hexdigest()
        assert reply['label'] == int(labels[i]) and reply['scores'] == scores[i].tolist()
        assert reply['sent_monotonic_ns'] <= reply['received_monotonic_ns']
        state = read_words(raw / f'native.snapshot_{i}.ram')
        before = initial if i == 0 else read_words(raw / f'native.snapshot_{i-1}.ram')
        assert state[2 + 128 * 36:] == initial[2 + 128 * 36:]
        if reply['label'] == 2:
            assert state == before and not reply['changed_records'] and reply['connection_bit_edits'] == 0
            queries += 1
        edits = sum((a ^ b).bit_count() for j, (a, b) in enumerate(zip(before[2:], state[2:])) if j % 36 >= 4)
        assert edits == reply['connection_bit_edits']
        checked.add(i)
    for event in events:
        if event['kind'] == 'preparation':
            check(event['response'])
        elif event['kind'] == 'response_pair':
            index = event['input_frame']; f = frame_events[index]; elapsed = f['elapsed']
            assert event['elapsed'] == elapsed
            previous = frames[by_elapsed[f['case'], max(0, elapsed - 8)]]
            sky = frames[by_elapsed[f['case'], 0]][50, 0].astype(np.int16)
            for action, reply in enumerate(event['replies']):
                check(reply)
                expected = encode(frames[index], previous, sky, action, 'projection', elapsed)
                assert np.array_equal(expected, inputs[reply['index']]), 'RGB/time feature source mismatch'
    assert native['samples'] == len(inputs)
    decisions = [e for e in events if e['kind'] == 'decision']
    feedback = [e for e in events if e['kind'] == 'feedback']
    contexts = json.loads((study / 'evidence/DEMO_PLAN.json').read_text())['contexts']
    for case, (decision, outcome, context) in enumerate(zip(decisions, feedback, contexts)):
        pair = next(e for e in events if e['kind'] == 'response_pair' and e['purpose'] == 'action_selection' and e['input_frame'] == decision['input_frame'])
        margins = [max(r['scores'][64:128]) - max(r['scores'][:64]) for r in pair['replies']]
        assert margins == decision['margins'] and decision['action'] == int(margins[1] > margins[0])
        assert decision['native_ids'] == [r['id'] for r in pair['replies']]
        assert max(r['received_monotonic_ns'] for r in pair['replies']) <= decision['mono_ns']
        action = decision['action']
        env = JoypadSpace(gym.make(f'SuperMarioBros-{context["level"]}-v0', render_mode='rgb_array'), SIMPLE_MOVEMENT)
        try:
            frame, info = env.reset(seed=82800)
            last = max(e['elapsed'] for e in frame_events.values() if e['case'] == case)
            start = context['wait'] + context['start']
            start_x = None; dead = False
            for elapsed in range(last + 1):
                assert np.array_equal(frame, frames[by_elapsed[case, elapsed]]), ('fresh rendered frame differs', case, elapsed)
                if elapsed == start:
                    start_x = int(info['x_pos'])
                if elapsed == last:
                    break
                motor = 0 if elapsed < context['wait'] else 1 if elapsed < start else 2 if 16 * action <= elapsed - start < 16 * action + 20 else 1
                frame, _, terminated, truncated, info = env.step(motor)
                dead = bool(info.get('death') or terminated or truncated)
                assert not dead or elapsed + 1 == last
            actual = bool(not dead and int(info['x_pos']) - start_x >= 70)
            assert outcome['success'] == actual and outcome['advance_evaluation_only'] == int(info['x_pos']) - start_x
        finally:
            env.close()
        reply = outcome['response']; check(reply)
        original_input = inputs[pair['replies'][action]['index']]
        assert np.array_equal(original_input, inputs[reply['index']]), 'feedback must teach its earlier decision input'
        assert reply['label'] == (2 if receipt['frozen'] else int(actual))
        assert outcome['input_frame'] == decision['input_frame']
    assert len(contexts) == len(decisions) == len(feedback) == 4
    assert checked == set(range(len(inputs)))
    result = dict(status='INDEPENDENT_LIVE_AUDIT_PASS', native=native, captured_frames=len(frames),
        protected_queries_unchanged=queries, fresh_rendered_frame_replay_exact=True,
        all_features_rebuilt_from_captured_RGB_and_observed_time=True, action_selection_from_measured_native_scores=True,
        outcomes=[dict(context=e['context'], action=e['action'], success=e['success'], connection_bit_edits=e['response']['connection_bit_edits']) for e in feedback],
        test_outcomes_collected=False, source_sha256=sha(__file__))
    save(capture / 'INDEPENDENT_AUDIT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('capture', type=Path)
    p.add_argument('raw', type=Path)
    a = p.parse_args()
    audit(a.study.resolve(), a.capture.resolve(), a.raw.resolve())
