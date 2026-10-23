"""A measured Mario pilot replay, including the harmful update and frozen controls."""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from visualize import BG, PANEL, INK, MUTED, BLUE, GOLD, font

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED = '#73d7b2', '#ef8584'


def render(fresh, video):
    z = json.loads((ROOT / 'results/2026-10-23-mario.json').read_text())
    receipt = json.loads((fresh / 'FRESH_EPISODES.json').read_text())
    assert hashlib.sha256((fresh / 'FRESH_EPISODES.json').read_bytes()).hexdigest() == z['fresh_receipt_sha256']
    clips = np.load(fresh / 'clips.npz', allow_pickle=False)
    for row in receipt['rows']:
        if 'clip_key' in row:
            assert hashlib.sha256(clips[row['clip_key']].tobytes()).hexdigest() == row['frames_sha256']
    base = Image.new('RGB', (3840, 2160), BG)
    d = ImageDraw.Draw(base)
    d.text((135, 70), 'AKD1500 / FIRST VISUAL-ACTION CONSEQUENCE SCREEN', font=font(34), fill=MUTED)
    d.text((135, 140), 'Learning helped here. It also hurt here.', font=font(82, True), fill=INK)
    d.text((135, 260), 'Fresh emulator episodes execute fixed action macros selected by measured native match scores.', font=font(36), fill=MUTED)
    d.text((180, 325), 'FROZEN / SAME INITIAL CONNECTIONS', font=font(31, True), fill=MUTED)
    d.text((1110, 325), 'AFTER NATIVE CONNECTION UPDATES', font=font(31, True), fill=BLUE)
    cases = [(81701, 108, 435), (81702, 116, 1275)]
    for seed, start, y in cases:
        for arm, x in [('frozen', 180), ('full', 1110)]:
            row = next(r for r in receipt['rows'] if r['seed'] == seed and r['start'] == start and r['arm'] == arm)
            d.rounded_rectangle((x - 20, y - 20, x + 788, y + 760), radius=22, fill=PANEL, outline='#34465a', width=3)
            color = GREEN if row['success'] else RED
            macro = 'JUMP NOW' if row['action'] == 0 else 'DELAY 16 FRAMES'
            d.text((x, y + 725), macro + (' / SAFE' if row['success'] else ' / FAIL'), font=font(29, True), fill=color)
        label = '81701 / start 108: corrects an unsuccessful delayed jump' if seed == 81701 else '81702 / start 116: changes a successful action into failure'
        d.text((180, y - 55), label, font=font(26), fill=MUTED)
    d.rounded_rectangle((2100, 345, 3705, 2035), radius=30, fill=PANEL, outline='#34465a', width=3)
    d.text((2160, 395), 'ALL EIGHT HELD-OUT STARTS / SAME LEVEL', font=font(32, True), fill=INK)
    for x, label in [(2160, 'Initialization'), (2570, 'Learned'), (2940, 'Frozen'), (3310, 'Action-only')]:
        d.text((x, 470), label, font=font(29), fill=MUTED)
    for i, row in enumerate(z['seed_results']):
        y = 545 + 90 * i
        d.text((2160, y), str(row['seed']), font=font(42, True), fill=INK)
        for x, key, color in [(2570, 'full', BLUE), (2940, 'frozen', MUTED), (3310, 'action_only', GOLD)]:
            d.text((x, y), f"{row[key]}/8", font=font(42, True), fill=color)
    d.text((2160, 860), '21/24 learned · 20/24 frozen · 18/24 constant', font=font(37, True), fill=INK)
    d.text((2160, 930), 'Only one extra success over frozen weights overall.', font=font(31), fill=MUTED)
    d.text((2160, 995), 'Two of three seeds pass the small predeclared gate.', font=font(31), fill=MUTED)
    d.text((2160, 1060), '72 fresh episodes matched the recorded consequences.', font=font(31), fill=MUTED)
    d.line((2160, 1150, 3640, 1150), fill='#34465a', width=3)
    d.text((2160, 1200), 'FINAL READBACK / SEED 81701', font=font(31, True), fill=BLUE)
    data = np.load(ROOT / 'results/2026-10-23-mario.npz', allow_pickle=False)
    initial = data['fixture__s81701_full__ram']
    final = data['s81701_full80__native_ram']
    changed = [not np.array_equal(initial[2 + j * 20:2 + (j + 1) * 20], final[2 + j * 20:2 + (j + 1) * 20]) for j in range(96)]
    for group, name in enumerate(['UNSAFE', 'SAFE', 'PROTECTED QUERY']):
        x = 2160 + group * 500
        d.text((x, 1280), name, font=font(27, True), fill=MUTED)
        for j in range(32):
            xx, yy = x + (j % 8) * 52, 1345 + (j // 8) * 52
            d.rounded_rectangle((xx, yy, xx + 34, yy + 34), radius=5, fill=GOLD if changed[group * 32 + j] else '#334457')
    d.text((2160, 1590), f'{sum(changed)} outcome records changed; query-class RAM stayed fixed.', font=font(30), fill=GOLD)
    d.text((2160, 1650), 'One square per configured neuron record; final RAM delta.', font=font(28), fill=MUTED)
    d.text((2160, 1730), 'Authored pixel encoder → native association memory', font=font(31), fill=INK)
    d.text((2160, 1790), '→ fixed score ranking → one of two supplied macros', font=font(31), fill=INK)
    d.text((2160, 1880), '12 experiences × 4 fixed epochs. No acting game RAM.', font=font(29), fill=MUTED)
    d.text((135, 2070), 'Recorded replay at 0.5× emulator time; held end frames. Raw RAM and scores measured; no electrical spike animation.', font=font(27), fill=MUTED)
    d.text((135, 2120), 'Small skill pilot, not full-game RL, learned perception, unseen-level transfer or CPU/energy advantage.', font=font(27), fill=MUTED)

    @functools.lru_cache(maxsize=16)
    def scaled(key, i):
        return Image.fromarray(clips[key][i]).resize((768, 720), Image.Resampling.NEAREST)

    def frame(k):
        image = base.copy()
        for seed, start, y in cases:
            for arm, x in [('frozen', 180), ('full', 1110)]:
                key = f's{seed}_{arm}_{start}'
                i = min(len(clips[key]) - 1, max(0, (k - 30) // 2))
                image.paste(scaled(key, i), (x, y))
        return image

    frame(220).save(ROOT / 'visual/mario-consequence-4k.png')
    if video:
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '3840x2160', '-r', '60', '-i', '-', '-an', '-c:v', 'h264_videotoolbox', '-b:v', '18M',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(ROOT / 'visual/mario-consequence-4k60.mp4')]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for k in range(720):
                process.stdin.write(frame(k).tobytes())
            process.stdin.close()
            assert process.wait() == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('fresh', type=Path)
    p.add_argument('--video', action='store_true')
    a = p.parse_args()
    render(a.fresh.resolve(), a.video)
