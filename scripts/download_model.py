#!/usr/bin/env python3
"""下载固定版本的基础模型快照，并逐个校验清单中记录的文件哈希。"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           'repro/model-manifest.json').read_text())
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=manifest['repo'], revision=manifest['revision'],
                          local_dir=str(args.output_dir),
                          allow_patterns=[x['path'] for x in manifest['files']])
    errors = []
    for expected in manifest['files']:
        path = args.output_dir / expected['path']
        if not path.is_file():
            errors.append('missing: ' + expected['path'])
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        if path.stat().st_size != expected['bytes'] or h.hexdigest() != expected['sha256']:
            errors.append('checksum mismatch: ' + expected['path'])
    print(json.dumps({'ok':not errors,'repo':manifest['repo'],
                      'revision':manifest['revision'],'files':len(manifest['files']),
                      'errors':errors},indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
