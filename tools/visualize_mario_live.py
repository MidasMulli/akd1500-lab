"""Wall-time-faithful game/AKD response film from independently audited live capture."""
from pathlib import Path
import argparse
import bisect
import functools
import hashlib
import json
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from visualize import BG, PANEL, INK, MUTED, BLUE, GOLD, font

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, PURPLE = '#73d7b2', '#ef8584', '#b59bea'
F = {n: font(n) for n in [23, 25, 27, 29, 31, 34, 38, 43, 52]}
B = {n: font(n, True) for n in [27, 31, 34, 38, 43, 52, 78]}


def render(video):
    data = json.loads((ROOT / 'visual/mario-live-data.json').read_text())
    confirmation = json.loads((ROOT / 'results/2026-10-23-mario-confirmation.json').read_text())
    assert confirmation['learning_effect_gate'] == 'FAIL'
    game_file = ROOT / 'visual/mario-live-game.npz'
    archive = ROOT / 'results/2026-10-23-mario-sealed.npz'
    assert hashlib.sha256(game_file.read_bytes()).hexdigest() == data['game_sha256']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == data['raw_archive_sha256']
    game = np.load(game_file, allow_pickle=False)
    images, clock = game['frames'], game['mono_offsets_ns']
    raw = np.load(archive, allow_pickle=False)
    states = raw['live_full01__snapshots']
    initial = raw['fixture__live_full01__ram']
    inputs = raw['fixture__live_full01__inputs'].reshape(-1, 1024)
    events = data['events']
    frame_events = {e['frame_id']: e for e in events if e['kind'] == 'frame'}
    packets = []
    for event in events:
        if event['kind'] == 'response_pair':
            for action, reply in enumerate(event['replies']):
                packets.append(dict(reply=reply, action=action, input_frame=event['input_frame'], purpose=event['purpose']))
        elif event['kind'] == 'feedback':
            packets.append(dict(reply=event['response'], action=event['action'], input_frame=event['input_frame'], purpose='teach_observed_outcome'))
    packets.sort(key=lambda e: e['reply']['received_monotonic_ns'])
    arrivals = [p['reply']['received_monotonic_ns'] for p in packets]
    pairs = [e for e in events if e['kind'] == 'response_pair']
    decisions = [e for e in events if e['kind'] == 'decision']
    feedback = [e for e in events if e['kind'] == 'feedback']
    base = Image.new('RGB', (3840, 2160), BG)
    d = ImageDraw.Draw(base)
    d.text((130, 65), 'AKD1500 / SYNCHRONIZED LIVE GAMEPLAY AND NATIVE RESPONSES', font=F[34], fill=MUTED)
    d.text((130, 133), 'Mario moves. The chip responds.', font=B[78], fill=INK)
    d.text((130, 245), 'Real device replies, native connection edits and game frames share a recorded wall-clock timeline.', font=F[38], fill=MUTED)
    d.text((130, 305), '56 captured frames/s · 8.4 query pairs/s · this 60 fps export preserves recorded pauses', font=F[29], fill=MUTED)
    d.rounded_rectangle((130, 385, 1710, 1880), radius=24, fill=PANEL, outline='#34465a', width=3)
    d.rounded_rectangle((1785, 385, 3710, 1585), radius=24, fill=PANEL, outline='#34465a', width=3)
    d.text((1835, 420), 'MEASURED NATIVE MATCH RESPONSES', font=B[38], fill=INK)
    d.text((1835, 482), f'One resident NPU {tuple(data["learning_np"])} · 192 records · scores 0–32', font=F[29], fill=MUTED)
    for group, (name, color, ids) in enumerate([('UNSAFE', RED, 'R0–63'), ('SAFE', GREEN, 'R64–127'), ('QUERY / PROTECTED', BLUE, 'R128–191')]):
        x = 1850 + 610 * group
        d.text((x, 555), name, font=B[27], fill=color)
        d.text((x, 600), ids + (' / not ranked' if group == 2 else ''), font=F[25], fill=MUTED)
        for j in range(64):
            xx, yy = x + (j % 8) * 65, 650 + (j // 8) * 50
            d.rounded_rectangle((xx, yy, xx + 45, yy + 39), radius=5, fill='#1e2e40', outline='#34465a', width=1)
    d.text((1835, 1150), 'RECORDED INPUT / LINK CHECK', font=B[31], fill=INK)
    d.text((1835, 1338), 'Blue: image feature · purple: observed-time feature · filled: matched', font=F[25], fill=MUTED)
    d.text((1835, 1385), 'Links are recorded RAM; gold marks inserted addresses. Response values above come from the chip.', font=F[25], fill=MUTED)
    d.line((1835, 1448, 3650, 1448), fill='#34465a', width=2)
    d.rounded_rectangle((130, 1910, 1710, 2040), radius=20, fill=PANEL)
    d.text((165, 1938), 'CPU: fixed RGB/time encoder, ranking and motor primitives.', font=F[29], fill=MUTED)
    d.text((165, 1988), 'AKD: native responses and outcome-driven connection updates.', font=F[29], fill=INK)
    d.rounded_rectangle((1785, 1620, 3710, 2040), radius=24, fill=PANEL)
    d.text((1835, 1650), 'SEPARATE SEALED CONFIRMATION / FIVE INITIALIZATIONS', font=B[31], fill=INK)
    for i, row in enumerate(confirmation['cohorts']):
        d.text((1835, 1700 + i * 43), f"{row['seed']}    learned {row['learned']}/128    frozen {row['frozen']}/128    constant 111/128", font=F[29], fill=MUTED)
    d.text((1835, 1935), '4/5 pair wins; +2.66 points. Preset learning-effect gate FAILED.', font=B[27], fill=GOLD)
    d.text((1835, 1985), 'Live movie: development cases. Table: separate held-out timing bank.', font=F[27], fill=MUTED)
    d.text((130, 2080), 'Color = match value · white = sampled rise · amber = RAM edit. Logical records; electrical spike timing is not measured.', font=F[29], fill=MUTED)

    def color(hex_value, value):
        rgb = tuple(int(hex_value[i:i+2], 16) for i in (1, 3, 5))
        return tuple(int(a * (.18 + .82 * value)) for a in rgb)

    @functools.lru_cache(maxsize=8)
    def resized(i):
        return Image.fromarray(images[i]).resize((1536, 1440), Image.Resampling.NEAREST)

    @functools.lru_cache(maxsize=64)
    def links(packet_index):
        packet = packets[packet_index]; reply = packet['reply']; i = reply['index']
        before = initial if i == 0 else states[i-1]
        update = packet['purpose'] == 'teach_observed_outcome' and bool(reply['changed_records'])
        unit = reply['changed_records'][0] if update else int(np.argmax(reply['scores'][:128]))
        state = states[i] if update else before
        words = state[2 + unit * 36 + 4:2 + (unit + 1) * 36]
        selected = [k for k in range(1024) if int(words[k // 32]) >> (k % 32) & 1]
        old_words = before[2 + unit * 36 + 4:2 + (unit + 1) * 36]
        added = {k for k in selected if not (int(old_words[k // 32]) >> (k % 32) & 1)}
        assert len(selected) == 32
        matched = [bool(inputs[i, k]) for k in selected]
        visual = sum(v and k % 512 < 256 for k, v in zip(selected, matched))
        timing = sum(v and k % 512 >= 256 for k, v in zip(selected, matched))
        if not update:assert visual + timing == reply['scores'][unit]
        return unit, selected, matched, visual, timing, added, update

    start = int(clock[0]); duration = (int(clock[-1]) - start) / 1e9
    def frame(k):
        offset = start + int(k / 60 * 1e9)
        wall = data['origin_lower_ns'] + offset
        index = max(0, min(len(images) - 1, int(np.searchsorted(clock, offset, side='right')) - 1))
        state = frame_events[index]; case = state['case']
        image = base.copy(); q = ImageDraw.Draw(image)
        image.paste(resized(index), (152, 412))
        q.text((130, 350), f'WORLD 1-1 / DEVELOPMENT CASE {case+1}/4 / FRAME {state["elapsed"]}', font=F[27], fill=INK)
        q.text((3050, 80), f'{min(k/60,duration):05.2f} s / recorded', font=F[29], fill=INK)
        current_feedback = [e for e in feedback if e['case'] == case and e['mono_ns'] <= wall]
        current_decisions = [e for e in decisions if e['case'] == case and e['mono_ns'] <= wall]
        if k/60 > duration:
            status, tone = 'CAPTURE COMPLETE / final readback held', MUTED
        elif current_feedback:
            status, tone = ('OUTCOME SAFE' if current_feedback[-1]['success'] else 'OUTCOME FAILED') + ' → NATIVE FEEDBACK', GREEN if current_feedback[-1]['success'] else RED
        elif current_decisions:
            action = current_decisions[-1]['action']
            status, tone = ('COMMITTED: JUMP NOW' if action == 0 else 'COMMITTED: WAIT 16 FRAMES, THEN JUMP'), BLUE
        else:
            status, tone = 'SUPPLIED PREFIX / native queries monitor the scene', MUTED
        q.text((1835, 1100), status, font=B[27], fill=tone)
        pindex = bisect.bisect_right(arrivals, wall) - 1
        if pindex >= 0:
            packet = packets[pindex]; reply = packet['reply']
            age = (wall - reply['received_monotonic_ns']) / 1e9
            source = frame_events[packet['input_frame']]
            if source['case'] == case:
                changed = set(reply['changed_records']) if age < .9 else set()
                for j, score in enumerate(reply['scores']):
                    group, local = divmod(j, 64); x = 1850 + 610 * group + (local % 8) * 65; y = 650 + (local // 8) * 50
                    tone = [RED, GREEN, BLUE][group]
                    rose = pindex > 0 and score > packets[pindex-1]['reply']['scores'][j] and age < .09
                    q.rounded_rectangle((x, y, x+45, y+39), radius=5, fill=color(tone, score / 32), outline=GOLD if j in changed else INK if rose else '#34465a', width=4 if j in changed else 2 if rose else 1)
                    q.rectangle((x+4, y+31, x+4+int(37*score/32), y+34), fill=tone)
                mode = 'TEACH PRIOR DECISION' if packet['purpose'] == 'teach_observed_outcome' else 'ACTION QUERY' if packet['purpose'] == 'action_selection' else 'SHADOW QUERY / DOES NOT REPLAN'
                q.text((1835, 1060), f'{mode} · A{packet["action"]} · packet {reply["id"]} · input frame {source["elapsed"]} · age {age:.2f}s', font=F[23], fill=GOLD if packet['purpose'] == 'teach_observed_outcome' else BLUE)
                if age < .05:
                    q.ellipse((3630, 424, 3662, 456), fill=BLUE)
                unit, selected, matched, visual, timing, added, update = links(pindex)
                title = f'RAM UPDATE R{unit}: {len(added)} new feature addresses / 32 selected links' if update else f'Max policy match R{unit}: {reply["scores"][unit]} = {visual} image + {timing} time'
                q.text((1835, 1210), title, font=B[31], fill=GOLD if update else INK)
                for i, (feature, active) in enumerate(zip(selected, matched)):
                    x = 1840 + i * 55; tone = BLUE if feature % 512 < 256 else PURPLE
                    q.rounded_rectangle((x, 1270, x+37, 1306), radius=5, fill=tone if active else '#1a2432', outline=GOLD if feature in added else tone, width=5 if feature in added else 2)
        completed = [e for e in feedback if e['response']['received_monotonic_ns'] <= wall]
        edits = sum(e['response']['connection_bit_edits'] for e in completed)
        q.text((1835, 1485), f'{len(completed)} observed {"outcome" if len(completed)==1 else "outcomes"} · {edits//2} native connection replacements', font=B[31], fill=GOLD)
        q.text((1835, 1535), 'Each edited record keeps 32 selected links. Protected queries leave RAM fixed.', font=F[25], fill=MUTED)
        return image

    first_update = feedback[0]['response']['received_monotonic_ns'] - data['origin_lower_ns'] - start
    frame(int((first_update / 1e9 + .08) * 60)).save(ROOT / 'visual/mario-live-responses-4k.png')
    if video:
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '3840x2160', '-r', '60', '-i', '-', '-an', '-c:v', 'h264_videotoolbox', '-b:v', '16M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(ROOT / 'visual/mario-live-responses-4k60.mp4')]
        child = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for k in range(math.ceil((duration+2)*60)):
                child.stdin.write(frame(k).tobytes())
            child.stdin.close(); assert child.wait() == 0
        finally:
            if child.poll() is None: child.terminate(); child.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--video', action='store_true'); a = p.parse_args(); render(a.video)
