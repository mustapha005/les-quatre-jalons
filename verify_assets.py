"""Verify the demo assets and original frozen model checksums."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'ASSET_MANIFEST.json').read_text())
for name,expected in manifest.items():
    p=root/name
    assert p.is_file(),f'Missing: {name}'
    assert p.stat().st_size==expected['bytes'],f'Wrong size: {name}'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],f'Changed: {name}'
for protocol in root.glob('artifacts/jalon*/frozen_protocol.json'):
    frozen=json.loads(protocol.read_text())
    for name,sha in frozen['weights'].items():
        p=protocol.parent/'models'/name/'best.keras'
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
print(f'OK: {len(manifest)} assets, model checksums match the original run.')
