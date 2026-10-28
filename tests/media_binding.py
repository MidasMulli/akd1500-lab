"""Check local sources or declared release digests; this does not download media.

Release payload sizes and digests are separately verified before publication.
The scientific numerical evidence remains locally checked in every test.
"""
import hashlib,json
from pathlib import Path

def binding_digest(root,path):
    root=Path(root);file=root/path
    if file.is_file():return hashlib.sha256(file.read_bytes()).hexdigest()
    if file.suffix!='.mp4':raise FileNotFoundError(file)
    manifest=json.loads((root/'docs/HISTORICAL_MEDIA_MANIFEST.json').read_text())
    matches=[x for x in manifest['media']if x['original_path']==path]
    if len(matches)!=1:raise FileNotFoundError(file)
    item=matches[0]
    if item['bytes']<=0 or len(item['sha256'])!=64:raise ValueError('Invalid archived media declaration')
    return item['sha256']
