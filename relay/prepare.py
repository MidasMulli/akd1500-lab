"""Compiler-only five-NPU graph. Reuses the reviewed intermediate-learner adapter."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import traceback
import numpy as np
import akida

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'tools'))
from prepare_network import Select, fc
from prepare_reduction import assemble
from program_codec import read, build, words
from relay.design import N, U, NW, CLASSES, THRESHOLD, SEEDS, initial, schedule


def message_weights():
    pairs = [(i, j) for i in range(CLASSES) for j in range(CLASSES) if i != j]
    w = np.zeros((U, len(pairs)), np.int8); thresholds = []
    for col, (i, j) in enumerate(pairs):
        w[i * 16:(i + 1) * 16, col] = 1
        w[j * 16:(j + 1) * 16, col] = -1
        thresholds.append(-1 if i < j else 0)
    expand = np.zeros((len(pairs), N), np.int8)
    for i in range(CLASSES): expand[i * 3:(i + 1) * 3, i * 128:(i + 1) * 128] = 1
    return w, np.asarray(thresholds), expand


def build_message(model):
    w, t, expand = message_weights()
    fc(model, 'pairwise_comparison', w, t)
    fc(model, 'categorical_message', expand, np.full(N, 2, np.int32))


def bind(prefixes, app):
    image, audit = assemble(prefixes[-1], app, N, U)
    h, data = read(image); data = bytearray(data); banks = []
    for index, prefix in enumerate(prefixes):
        si, sd = read(prefix); lr = si['learning']
        ram = words(sd, lr['ram'][0]); inf = words(sd, lr['inference']); lea = words(sd, lr['learning'])
        if index == len(prefixes) - 1:
            ni = words(data, h['learning']['inference']); nl = words(data, h['learning']['learning'])
        else:
            candidates = [(ri, words(data, sp)) for ri, rec in enumerate(h['passes'][0]) for sp in rec
                          if len(words(data, sp)) == 52 + len(ram)
                          and words(data, sp)[0] == inf[0] and words(data, sp)[52:54] == ram[:2]]
            assert len(candidates) == 1, ('ambiguous learner binding', index, len(candidates))
            ri, packet = candidates[0]; ni = packet[:52]; nl = ni[:]
            assert all(j in [46, 47, 49] for j, (a, b) in enumerate(zip(inf, ni)) if a != b)
            assert all(j >= 2 and (j - 2) % 20 in [1, 2, 3]
                       for j, (a, b) in enumerate(zip(ram, packet[52:])) if a != b)
            for j in [38, 39, 40]: assert ni[j] == inf[j]; nl[j] = lea[j]
            sp = [len(data), len(ni) + len(ram)]
            data.extend(struct.pack('<' + 'I' * (len(ni) + len(ram)), *(ni + ram)))
            h['passes'][0][ri] = [sp]
        banks.append(dict(np=lr['np'], ram=ram, infer=ni, learn=nl))
    struct.pack_into('<I', data, 0, len(data) - 4)
    result = build(h, bytes(data)); hh, dd = read(result); assert hh == h and dd == bytes(data)
    fixed = []
    for rec in h['passes'][0]:
        for sp in rec:
            packet = words(data, sp); at = 0
            while at < len(packet):
                header, size = packet[at:at + 2]; length = (size >> 16) & 0x3fff
                if size >> 31: length *= 16
                t = packet[at:at + 2 + length]; assert len(t) == 2 + length; at += len(t)
                mask, uid = (header >> 8) & 15, header & 15
                if mask and uid in [0, 2, 4, 6]:
                    assert not mask & (mask - 1)
                    npid = [header >> 24, (header >> 16) & 255, mask.bit_length() - 1]
                    if npid not in [b['np'] for b in banks]: fixed.append(dict(np=npid, packet=t))
            assert at == len(packet)
    assert len(fixed) == 3 and len({tuple(f['np']) for f in fixed}) == 3
    return result, banks, fixed, audit


def prepare(study, output='fixtures-relay01'):
    out = study / output; out.mkdir(exist_ok=False)
    x, _ = schedule('recognition_repair'); rows = []
    # Separate virtual truth-table verification of native categorical communication.
    vm = akida.Model(); vm.add(akida.InputData(input_shape=(1, 1, U), input_bits=1)); build_message(vm)
    a = np.zeros((128, U), np.uint8); rng = np.random.default_rng(97401)
    counts = rng.integers(0, 17, (128, 4)); counts[:4] = 0
    for i, group in enumerate(counts):
        for j, count in enumerate(group): a[i, j * 16:j * 16 + count] = 1
    got = vm.forward(a.reshape(-1, 1, 1, U)).reshape(-1, N)
    want = np.zeros_like(got)
    for i, group in enumerate(counts): c = int(group.argmax()); want[i, c * 128:(c + 1) * 128] = 1
    assert np.array_equal(got, want), 'categorical messaging virtual truth table'
    for seed in SEEDS:
        name = f's{seed}'; m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, N), input_bits=1))
        prefixes, ws = [], []
        try:
            for index, ram in enumerate(initial(seed)):
                if index: build_message(m)
                layer_name = 'recognition' if index == 0 else 'routing'
                m.add(akida.FullyConnected(name=layer_name, units=U, weights_bits=1, activation=True, act_bits=1))
                m.compile(optimizer=akida.AkidaUnsupervised(num_weights=NW, num_classes=CLASSES,
                          initial_plasticity=.5, min_plasticity=.25, plasticity_decay=.125, learning_competition=.25))
                l = m.get_layer(layer_name); w = np.zeros((N, U), np.int8)
                for j in range(U):
                    for k in range(N): w[k, j] = (ram[2 + j * 20 + 4 + k // 32] >> (k % 32)) & 1
                l.set_variable('weights', w.reshape(1, 1, N, U))
                l.set_variable('threshold_learning', np.zeros(U, np.int32))
                l.set_variable('threshold', np.full(U, THRESHOLD, np.int32))
                l.set_variable('act_step', np.ones(U, np.float32)); ws.append(w)
                m.map(akida.AKD1500(), hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(akida.AKD1500()))
                assert len(m.sequences) == 1 and len(m.sequences[0].passes) == 1
                prefixes.append(bytes(m.sequences[0].program))
            readout = np.zeros((U, CLASSES), np.int8)
            for j in range(U): readout[j, j // 16] = 1
            fc(m, 'destination_support', readout, None, activation=False)
            # Include inputs activating each recognition class, in addition to task inputs.
            test = np.vstack([x[:16], ws[0][:, [0, 16, 32, 48]].T]).astype(np.uint8)
            v = (test.astype(np.int32) @ ws[0].astype(np.int32) > THRESHOLD).reshape(-1, 4, 16).sum(axis=2)
            msg = np.zeros((len(test), N), np.int32)
            for i, c in enumerate(v.argmax(axis=1)): msg[i, c * 128:(c + 1) * 128] = 1
            expected = (msg @ ws[1].astype(np.int32) > THRESHOLD).reshape(-1, 4, 16).sum(axis=2)
            m.map(None); actual = m.forward(test.reshape(-1, 1, 1, N)).reshape(-1, CLASSES)
            assert np.array_equal(expected, actual), 'whole virtual graph'
            m.map(akida.AKD1500(), hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(akida.AKD1500()))
            assert len(m.sequences) == 1 and len(m.sequences[0].passes) == 1
            app = bytes(m.sequences[0].program); (out / (name + '.compiler.bin')).write_bytes(app)
            image, banks, fixed, audit = bind(prefixes, app)
            layout = [f'{N} {U} {CLASSES} 2 3 {THRESHOLD}']
            for k, bank in enumerate(banks):
                for ext in ['ram', 'infer', 'learn']:
                    values = bank[ext]; (out / f'{name}.bank{k}.{ext}').write_bytes(struct.pack('<'+'I'*len(values), *values))
                layout.append(' '.join(map(str, bank['np'] + [len(bank['ram'])])))
            for k, packet in enumerate(fixed):
                values = packet['packet']; (out / f'{name}.fixed{k}.ram').write_bytes(struct.pack('<'+'I'*len(values), *values))
                layout.append(' '.join(map(str, packet['np'] + [len(values) - 2])))
            (out / (name + '.layout')).write_text('\n'.join(layout) + '\n')
            (out / (name + '.bin')).write_bytes(image)
            mapping = [dict(layer=l.name, nps=[dict(type=str(q.type), np=[q.ident.col, q.ident.row, q.ident.id])
                                              for q in l.mapping.nps]) for l in m.sequences[0].passes[0].layers]
            plan = dict(seed=seed, banks=[b['np'] for b in banks], mapping=mapping,
                        fixed=[f['np'] for f in fixed], inputs=N, neurons=U, classes=CLASSES, threshold=THRESHOLD,
                        program_sha256=hashlib.sha256(image).hexdigest(), adapter=audit)
            (out / (name + '.json')).write_text(json.dumps(plan, indent=2) + '\n')
            row = dict(status='COMPILER_AND_VIRTUAL_PASS', **plan)
        except Exception as e:
            row = dict(seed=seed, status='FAILED_PRESERVED', error=str(e), traceback=traceback.format_exc())
        rows.append(row); (out / 'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY',
                    physical_device_opened=False, message_truth_table_cases=128, cells=rows), indent=2) + '\n')
        print(json.dumps(row), flush=True)
    assert all(r['status'] == 'COMPILER_AND_VIRTUAL_PASS' for r in rows), 'compiler gate failed'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path)
    p.add_argument('--output', default='fixtures-relay01'); a = p.parse_args(); prepare(a.study, a.output)
