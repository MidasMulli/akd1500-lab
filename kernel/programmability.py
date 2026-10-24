"""Read-only source inventory; absence is scoped, not a silicon-impossibility proof."""
from pathlib import Path
import argparse
import hashlib
import json
import re

REVIEWED = {
    'api/akida/hardware_device.h': ['toggle_learn', 'learn_mem', 'update_learn_mem', 'program'],
    'api/akida/program_info.h': ['serialized_program', 'can_learn', 'number_of_passes'],
    'src/device_programmer.cpp': ['program_single_pass', 'program_record', 'program_track', 'configure_learning_mode_single_pass'],
    'src/hardware_device_impl.cpp': ['toggle_learn', 'pipeline', 'update_learn_mem'],
    'src/dma_engine.cpp': ['toggle_pipeline'],
}


def audit(runtime):
    files = [p for p in runtime.rglob('*') if p.is_file() and p.suffix in ['.cpp', '.h', '.hpp']]
    inventory = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    reviewed, hits = [], {}
    for rel, symbols in REVIEWED.items():
        p = runtime / rel; text = p.read_text(); lines = text.splitlines()
        found = {s: [i + 1 for i, line in enumerate(lines) if s in line] for s in symbols}
        assert all(found.values()), ('source API changed', rel, found)
        reviewed.append(dict(path=rel, sha256=inventory[rel], symbols=found))
    for term in ['microcode', 'firmware', 'opcode', 'instruction']:
        hits[term] = []
        for p in files:
            lines = p.read_text(errors='replace').splitlines()
            matches = [i + 1 for i, line in enumerate(lines) if re.search(r'\b' + term + r'\b', line, re.I)]
            if matches: hits[term].append(dict(path=str(p.relative_to(runtime)), lines=matches))
    return dict(status='READ_ONLY_SOURCE_AUDIT', runtime_source_files=len(files), source_inventory=inventory,
        reviewed_entrypoints=reviewed, keyword_locations=hits,
        disposition={
            'host_runtime': 'User-authored transport/runtime and known descriptor/configuration tracks are exercised',
            'native_graph': 'Compiler-generated graph/weights/routing can be composed with checked adapters',
            'learning_configuration': 'Native learning mode and learning RAM are exposed; optimizer parameters configure the built-in rule',
            'arbitrary_learning_instruction_upload': 'Not established in the reviewed API/source; this is not proof of physical impossibility',
            'fixed_logic_vs_internal_microcode': 'Unresolved by this source audit',
            'akidatag_firmware': 'Official AkidaTag firmware targets an external Nordic nRF5340 host; it does not establish AKD learning-engine replacement'},
        primary_sources=[
            'https://doc.brainchipinc.com/api_reference/akida_apis.html',
            'https://doc.brainchipinc.com/examples/edge/plot_2_edge_learning_parameters.html',
            'https://github.com/Brainchip-Inc/akida_examples/blob/main/doc/user_guide/engine.rst',
            'https://github.com/Brainchip-Inc/AkidaTag',
            'https://github.com/Brainchip-Inc/AkidaTag/blob/main/src/deps/VENDORING.md'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('runtime', type=Path); p.add_argument('output', type=Path); a = p.parse_args()
    r = audit(a.runtime); a.output.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(dict(status=r['status'], files=r['runtime_source_files'], keyword_locations=r['keyword_locations'])))
