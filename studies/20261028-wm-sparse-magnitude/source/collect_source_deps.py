"""Read-only local source closure, excluding SDK/library redistribution."""
from pathlib import Path
import ast,hashlib,json,re,zipfile
B=Path(__file__).resolve().parents[1];lab=B.parents[1];roots=[B/'source',lab/'visual_memory',lab/'src',lab,lab/'tools'];todo=list((B/'source').glob('*.py'))+[B/'source/sparse_actor.cpp'];seen=set();owned={};external=set()
while todo:
    p=todo.pop().resolve()
    if p in seen:continue
    seen.add(p);text=p.read_text();rel=str(p.relative_to(lab));owned[rel]=p
    if p.suffix=='.py':
        try:tree=ast.parse(text)
        except SyntaxError:continue
        for node in ast.walk(tree):
            names=[node.module]if isinstance(node,ast.ImportFrom)and node.module else[a.name for a in node.names]if isinstance(node,ast.Import)else[]
            for name in names:
                local=[r/(name.replace('.','/')+'.py')for r in [p.parent,lab,lab/'tools']];match=next((x for x in local if x.is_file()),None)
                if match is not None:todo.append(match)
                else:external.add(name.split('.')[0])
    else:
        for name in re.findall(r'^\s*#include\s+"([^"]+)"',text,re.M):
            match=next((r/name for r in [p.parent]+roots if(r/name).is_file()),None)
            if match is not None:todo.append(match)
            else:external.add(name)
dest=B/'OWNED_SOURCE_DEPS.zip';assert not dest.exists();receipt=dict(status='EXACT_OWNED_SOURCE_DEPENDENCY_CLOSURE',files={rel:hashlib.sha256(p.read_bytes()).hexdigest()for rel,p in sorted(owned.items())},external_dependencies=sorted(external),not_included='Licensed Akida SDK headers/library, compiler, Python/Node/Chrome/FFmpeg; installed versions and actor/program hashes are in receipts.')
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED)as z:
    for rel,p in sorted(owned.items()):z.write(p,rel)
with zipfile.ZipFile(dest)as z:
    assert z.testzip()is None
    for rel,h in receipt['files'].items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
receipt['archive_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();(B/'OWNED_SOURCE_DEPS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(files=len(owned),external=receipt['external_dependencies'],sha256=receipt['archive_sha256'])))
