#!/usr/bin/env python3
"""Verify a curated documentation/source bundle without network or optional dependencies."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
manifest=json.loads((HERE/'bundle-manifest.json').read_text())
for row in manifest['files']:
    p=(ROOT/row['path']).resolve()
    assert p.is_relative_to(ROOT.resolve()), row['path']
    assert p.is_file(), row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'], row['path']
ledger=json.loads((HERE/'incidents.json').read_text())
ids=[c['id'] for c in ledger['incidents']]
assert len(ids)==len(set(ids))==manifest['incident_count']
for c in ledger['incidents']:
    for k in ['symptom','diagnosis','action','result','remaining_limit','status']:
        assert isinstance(c[k],str) and c[k].strip(),(c['id'],k)
    for p in c['evidence']+c['implementation']:
        assert (ROOT/p).is_file(),(c['id'],p)
    assert f'id="{c["id"].lower()}"' in (HERE/'RETROSPECTIVE.md').read_text()
print(json.dumps({'status':'passed','project':manifest['project'],'incidents':len(ids),'files':len(manifest['files']),'model_requests':0,'scope':'File integrity and documentation references only'},ensure_ascii=False))
