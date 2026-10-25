"""Sealed software counterexamples for the qualified repetition-admission circuit.

No SDK, device, compiler, host action policy, or physical launch is used here.
Reference predictions on new schedules remain forecasts of hardware behavior.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from admission import learned as circuit
from admission import model

ROOT = Path(__file__).resolve().parents[1]
CASES = ("original", "repeated_noise_spaced_signal", "repeated_noise_repeated_signal")
ARMS = ("native", "open", "frozen", "manager_frozen")
ARRAYS = ("inputs", "outputs", "cpu", *(
    f"{name}{k}" for k in range(3) for name in ("states", "potentials", "features")
))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(case, arm):
    if case == "original":
        return circuit.schedule("heldout", arm)
    observations = []

    def add(keys, phase, teach=True):
        observations.extend(dict(key=int(key), phase=phase, teach=teach) for key in keys)

    add([0] * 12 + [1] * 12, "bootstrap")
    add([0, 1] * 2, "old_probe", False)
    add([key for key in range(2, 8) for _ in range(16)], "distractors")
    if case == "repeated_noise_spaced_signal":
        for repeat in range(8):
            for key in range(8, 12):
                add([key], "new_useful")
                noise = 2 + (repeat + key) % 6
                add([noise, noise], "distractors")
    elif case == "repeated_noise_repeated_signal":
        add([key for key in range(8, 12) for _ in range(16)], "new_useful")
    else:
        raise ValueError(case)
    add([0, 1, 8, 9, 10, 11] * 4, "final_probe", False)
    add(np.random.default_rng(114901).choice([0, 1, 8, 9, 10, 11], 96), "heldout", False)
    result = []
    for observation, obs in enumerate(observations):
        base = dict(obs, observation=observation)
        learn_manager = not (arm == "manager_frozen" and obs["phase"] != "bootstrap")
        result.append(dict(base, index=len(result), teacher=False, label=-1,
                           manager_learn=learn_manager,
                           labels=[0 if learn_manager else -1, -1, -1]))
        if obs["teach"]:
            label = model.OUTCOME[obs["key"]]
            if arm == "frozen" and obs["phase"] != "bootstrap":
                label = -1
            result.append(dict(base, index=len(result), teacher=True, label=label,
                               manager_learn=False, labels=[-1, label, label]))
    assert len(result) <= 1024
    return result


def masks(ram, n, u):
    words = np.asarray(ram, dtype="<u4")[2:].reshape(u, 4 + n // 32)[:, 4:]
    return np.unpackbits(words.view("u1"), axis=1, bitorder="little")


def verify(a, initial):
    report = {"status": "SOFTWARE_INVARIANTS_PASS", "banks": []}
    for k, (n, u, _) in enumerate(circuit.DIMENSIONS):
        states = a[f"states{k}"]
        before = np.asarray(initial[k], "<u4")
        plastic = [0] if k == 0 else model.P0 if k == 1 else model.P1
        fixed = [j for j in range(u) if j not in plastic]
        initial_records = before[2:].reshape(u, 4 + n // 32)
        blocked = admitted = replacements = 0
        for i, e in enumerate(a["events"]):
            after = states[i]
            record = after[2:].reshape(u, 4 + n // 32)
            assert np.array_equal(record[fixed], initial_records[fixed]), "fixed helpers changed"
            current_masks = masks(after, n, u)
            assert np.all(current_masks.sum(1) == 32), "connection budget"
            if e["labels"][k] < 0:
                assert np.array_equal(after, before), "disabled bank changed"
            if k == 2 and e["teacher"] and e["label"] >= 0:
                if a["features2"][i].any():
                    admitted += 1
                else:
                    blocked += 1
                    assert np.array_equal(after, before), "blocked teacher changed memory"
            old_masks = masks(before, n, u)
            replacements += int(((current_masks == 1) & (old_masks == 0)).sum())
            before = after
        report["banks"].append(dict(bank=k, admitted_teachers=admitted,
                                    blocked_teachers=blocked,
                                    connection_replacements=replacements))
    return report


def counts(a, keys=None):
    result = dict(correct=0, missing=0, ambiguous=0, wrong=0, total=0)
    for e in a["events"]:
        if e["teacher"] or e["phase"] != "heldout" or (keys is not None and e["key"] not in keys):
            continue
        prediction = a["outputs"][e["index"], 64:72].reshape(2, 4).any(1)
        actual = model.OUTCOME[e["key"]]
        result["total"] += 1
        category = ("ambiguous" if prediction.all() else "missing" if not prediction.any()
                    else "correct" if prediction[actual] else "wrong")
        result[category] += 1
    return result


def run(seeds, output):
    output = Path(output)
    assert not output.exists() and not output.with_suffix(".npz").exists(), "preserve earlier attempts"
    sources = [Path(__file__), ROOT / "admission/learned.py", ROOT / "admission/model.py",
               ROOT / "pong/relearn/model.py",
               ROOT / "studies/20261025-admission-relevance/PROTOCOL.md"]
    report = dict(kind="SOFTWARE_REPLAY_ONLY", physical_sessions=0, status="RUNNING",
                  seeds=seeds, source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
                  plans={f"{case}/{arm}": rows(case, arm) for case in CASES for arm in ARMS}, runs=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    arrays = {}
    started = time.monotonic()
    try:
        for seed in seeds:
            initial = [circuit.packed(circuit.weights(seed, k), k) for k in range(3)]
            for case in CASES:
                for arm in ARMS:
                    assert time.monotonic() - started < 120, "bounded software screen"
                    a = circuit.simulate(initial, "heldout", arm, events=report["plans"][f"{case}/{arm}"])
                    checks = verify(a, initial)
                    assessment = circuit.assess(a)
                    if case == "original" and arm == "native":
                        assert assessment["heldout"]["protected"]["correct"] == 96
                    if case == "original" and seed == 114101:
                        assert assessment["heldout"]["protected"]["correct"] == (96 if arm == "native" else 30)
                    name = f"s{seed}_{case}_{arm}"
                    for key in ARRAYS:
                        arrays[f"{name}__{key}"] = a[key]
                    entry = dict(seed=seed, case=case, arm=arm, events=len(a["events"]),
                                 heldout=counts(a), old_heldout=counts(a, {0, 1}),
                                 later_heldout=counts(a, {8, 9, 10, 11}),
                                 assessment=assessment, verification=checks)
                    report["runs"].append(entry)
                    print(json.dumps({key: entry[key] for key in ("seed", "case", "arm", "heldout", "later_heldout")}), flush=True)
        report["status"] = "SOFTWARE_REPLAY_PASS"
    except Exception as exc:
        report.update(status="FAILED_PRESERVED", failure=repr(exc))
        raise
    finally:
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
        report["arrays_sha256"] = digest(output.with_suffix(".npz"))
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")


def audit(output):
    output = Path(output)
    report = json.loads(output.read_text())
    assert report["status"] == "SOFTWARE_REPLAY_PASS"
    assert digest(output.with_suffix(".npz")) == report["arrays_sha256"]
    for path, expected in report["source_hashes"].items():
        assert digest(ROOT / path) == expected, ("source drift", path)
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as saved:
        for entry in report["runs"]:
            seed, case, arm = (entry[k] for k in ("seed", "case", "arm"))
            initial = [circuit.packed(circuit.weights(seed, k), k) for k in range(3)]
            assert report["plans"][f"{case}/{arm}"] == rows(case, arm)
            a = circuit.simulate(initial, "heldout", arm, events=report["plans"][f"{case}/{arm}"])
            prefix = f"s{seed}_{case}_{arm}__"
            for key in ARRAYS:
                assert np.array_equal(saved[prefix + key], a[key]), (prefix, key)
            assert counts(a) == entry["heldout"]
            assert counts(a, {0, 1}) == entry["old_heldout"]
            assert counts(a, {8, 9, 10, 11}) == entry["later_heldout"]
            assert circuit.assess(a) == entry["assessment"]
            assert verify(a, initial) == entry["verification"]
    print(json.dumps(dict(status="SAVED_SOFTWARE_REPLAY_PASS", runs=len(report["runs"]), physical_sessions=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "audit"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, choices=model.SEEDS, default=[model.SEEDS[0]])
    args = parser.parse_args()
    if args.mode == "run":
        run(args.seeds, args.output)
    else:
        audit(args.output)
