"""Measured sparse-work curves and RAM edits; no inferred physical spike traces."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from metal.sparse import audit_public as sparse_audit
from metal.interference import audit_public as coupled_audit

P = Path(__file__).resolve().parents[1]
BG, PANEL, INK, MUTED = '#09131f', '#101f30', '#edf4fc', '#a6b8cc'
BLUE, GOLD, PURPLE = '#72caff', '#ffc17d', '#c7b2ff'


def main():
    sparse_audit(); coupled_audit()
    sparse = json.loads((P / 'results/2026-10-23-sparse-work.json').read_text())
    coupled = json.loads((P / 'results/2026-10-23-learning-interference.json').read_text())
    sr = {r['name']: r for r in sparse['runs']}; cr = {r['name']: r for r in coupled['runs']}
    def cycles(n, u, nw): return sr[f'screen_n{n}_u{u}_w{nw}_compact']['audit']['groups'][0]['median_cycles']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#3b526b'})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
    fig.text(.06, .945, 'AKD1500 / sparse state, measured work', fontsize=29, weight='bold')
    fig.text(.06, .90, '70 native sessions  ·  19,712 forwards  ·  27,136 complete bank-state captures  ·  one physical chip', fontsize=13, color=MUTED)
    rects = [[.075, .555, .375, .245], [.565, .555, .36, .245], [.075, .18, .375, .245], [.565, .18, .36, .245]]
    axes = [fig.add_axes(rect, facecolor=PANEL) for rect in rects]
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.set_axisbelow(True)
        ax.grid(axis='y', color='#30455b', alpha=.6)
    a, b, c, d = axes
    a.set_title('01  Fewer connections did not make queries cheaper', loc='left', pad=22, weight='bold', fontsize=14)
    for n, color, marker in [(128, BLUE, 'o'), (512, GOLD, 's'), (1024, PURPLE, '^')]:
        vals = [cycles(n, 64, nw) for nw in [8, 32, 64]]
        a.plot([8, 32, 64], vals, marker=marker, color=color, lw=2.2, label=f'{n:,} input bits')
    a.set(xlim=(3, 69), ylim=(0, 11700), xticks=[8, 32, 64], xlabel='Selected connections per neuron record', ylabel='Initial-query DMA monitor cycles / input')
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k'))
    a.legend(loc='lower right', frameon=False, labelcolor=INK, fontsize=10)
    a.text(.025, .94, '64 records · same placement · fixed two-count output', transform=a.transAxes, va='top', fontsize=10, color=MUTED)
    b.set_title('02  Larger input records added service cost', loc='left', pad=22, weight='bold', fontsize=14)
    n = [128, 512, 1024]; measured = [(cycles(v, 256, 32) - cycles(v, 64, 32)) / 192 for v in n]
    b.plot(n, measured, '-o', lw=2.3, color=BLUE, label='Measured increment / added record')
    b.plot(n, [4 + v // 32 for v in n], '--s', lw=1.6, color=GOLD, label='Words stored per record (reference)')
    for v, z in zip(n, measured): b.annotate(f'{z:.2f}', (v, z), (5, 9), textcoords='offset points', fontsize=11)
    b.set(xlim=(60, 1120), ylim=(0, 45), xticks=n, xlabel='Binary input dimensions', ylabel='Added monitor cycles / added neuron record')
    b.legend(loc='upper left', frameon=False, labelcolor=INK, fontsize=9)
    b.text(.025, .04, 'Consistent with record scanning; does not reveal the exact circuit.', transform=b.transAxes, fontsize=9, color=MUTED)
    c.set_title('03  Both plastic banks edited memory in one job', loc='left', pad=22, weight='bold', fontsize=14)
    with np.load(P / 'results/2026-10-23-learning-interference.npz', allow_pickle=False) as z:
        for bank, offset, color in [(0, 0, BLUE), (1, 72, GOLD)]:
            states = z[f'full_s97101_m3__bank{bank}.snapshots']
            before = states[511:639, 2:].reshape(128, 64, 20)
            after = states[512:640, 2:].reshape(128, 64, 20)
            events, records = np.nonzero(np.any(before != after, axis=2))
            c.scatter(events + 1, records + offset, s=16, marker='s', color=color, linewidths=0)
    c.axhline(67, color='#53697e', lw=.8)
    c.set(xlim=(0, 129), ylim=(-7, 160), yticks=[0, 32, 63, 72, 104, 135], yticklabels=['A00', 'A32', 'A63', 'B00', 'B32', 'B63'], xlabel='Native test job after replaying the common training prefix', ylabel='Neuron record whose RAM changed')
    c.text(.035, .94, 'Actual captured RAM edits · seed 97101', transform=c.transAxes, fontsize=10, color=MUTED)
    d.set_title('04  Mixed teaching preserved each solo state', loc='left', pad=22, weight='bold', fontsize=14)
    edits = np.asarray([[sum(cr[f'full_s{seed}_m{m}']['audit']['test_bank_edit_events'][k] for seed in [97101, 97102, 97103]) for k in range(2)] for m in range(4)])
    x = np.arange(4); d.bar(x - .17, edits[:, 0], width=.32, color=BLUE, label='Bank A')
    d.bar(x + .17, edits[:, 1], width=.32, color=GOLD, label='Bank B')
    for i, row in enumerate(edits):
        for dx, value in zip([-.17, .17], row): d.text(i + dx, value + 8, str(int(value)), ha='center', fontsize=11)
    d.set(ylim=(0, 330), xticks=x, xticklabels=['Neither', 'A only', 'B only', 'Both'], xlabel='Teaching enabled · shared teacher labels', ylabel='Native test jobs with a bank-state edit')
    d.legend(loc='upper right', frameon=False, labelcolor=INK, fontsize=10)
    d.text(.025, .94, '3 seeds × 128 test jobs per mask', transform=d.transAxes, fontsize=10, color=MUTED)
    fig.text(.06, .085, 'Querying, learning and transport remain separate measurements. Each captured solo/mixed bank state matched exactly.', fontsize=12)
    fig.text(.06, .045, 'DMA service ≠ arithmetic-only cycles. Same-job edits ≠ clock-for-clock overlap. Shared teacher, feed-forward graph; no CPU or energy advantage established.', fontsize=10, color=MUTED)
    png = P / 'visual/execution-limits-4k.png'; svg = P / 'visual/execution-limits.svg'
    fig.savefig(png, dpi=200, facecolor=BG); fig.savefig(svg, facecolor=BG); plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    table = P / 'visual/execution-limits-data.csv'
    with table.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['seed', 'teaching_mask', 'test_jobs', 'bank_A_edited_jobs', 'bank_B_edited_jobs', 'same_job_both_edited', 'median_DMA_cycles'])
        for seed in [97101, 97102, 97103]:
            for m in range(4):
                a = cr[f'full_s{seed}_m{m}']['audit']; w.writerow([seed, m, a['test_events'], *a['test_bank_edit_events'], a['same_event_both_banks_edited'], a['test_median_cycles']])
    paths = ['results/2026-10-23-sparse-work.json', 'results/2026-10-23-sparse-work.npz',
             'results/2026-10-23-learning-interference.json', 'results/2026-10-23-learning-interference.npz',
             'tools/visualize_execution_limits.py', 'visual/execution-limits-4k.png', 'visual/execution-limits.svg', 'visual/execution-limits-data.csv']
    binding = dict(status='BOUND_TO_RECOUNTED_NATIVE_OBSERVATIONS', hashes={p: hashlib.sha256((P / p).read_bytes()).hexdigest() for p in paths})
    (P / 'visual/EXECUTION_LIMITS_SOURCE_BINDING.json').write_text(json.dumps(binding, indent=2) + '\n')
    print(png)


if __name__ == '__main__': main()
