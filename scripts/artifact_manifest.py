#!/usr/bin/env python3
"""生成并验证文件及 tar 归档的逐项 SHA-256 清单，用于检查备份是否完整。"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def entry(path, root):
    name = path.relative_to(root).as_posix()
    if path.is_symlink():
        return {'path': name, 'type': 'symlink', 'target': os.readlink(path)}
    if path.is_file():
        with path.open('rb') as stream:
            return {'path': name, 'type': 'file', 'bytes': path.stat().st_size,
                    'sha256': digest(stream)}
    return None


def inventory(root):
    for path in sorted(root.rglob('*')):
        item = entry(path, root)
        if item is not None:
            yield item


def verify(expected, observed):
    seen, errors = set(), []
    for item in observed:
        name = item['path']
        if name in seen:
            errors.append('duplicate: ' + name)
        seen.add(name)
        if expected.get(name) != item:
            errors.append('changed or unexpected: ' + name)
    errors.extend('missing: ' + name for name in sorted(set(expected) - seen))
    report = {'ok': not errors, 'expected_entries': len(expected),
              'verified_entries': len(seen), 'errors': errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def archive_entries(archive, prefix):
    prefix_parts = PurePosixPath(prefix).parts
    with tarfile.open(archive, mode='r|*') as source:
        for member in source:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('unsafe archive member: ' + member.name)
            if member.isdir():
                continue
            if prefix_parts and p.parts[:len(prefix_parts)] != prefix_parts:
                raise ValueError('unexpected archive prefix: ' + member.name)
            name = PurePosixPath(*p.parts[len(prefix_parts):]).as_posix()
            if member.issym():
                yield {'path': name, 'type': 'symlink', 'target': member.linkname}
            elif member.isfile():
                with source.extractfile(member) as stream:
                    yield {'path': name, 'type': 'file', 'bytes': member.size,
                           'sha256': digest(stream)}
            else:
                raise ValueError('unsupported archive member: ' + member.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('root', type=Path)
    create.add_argument('--output', required=True, type=Path)
    check = sub.add_parser('verify')
    check.add_argument('root', type=Path)
    check.add_argument('--manifest', required=True, type=Path)
    check_tar = sub.add_parser('verify-archive')
    check_tar.add_argument('archive', type=Path)
    check_tar.add_argument('--prefix', default='')
    check_tar.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'create':
        root = args.root.resolve()
        if not root.is_dir():
            parser.error('root must be a directory')
        if args.output.resolve().is_relative_to(root):
            parser.error('write the manifest outside the inventoried directory')
        items = list(inventory(root))
        report = {'schema_version': 1, 'entries': items,
                  'file_count': sum(x['type'] == 'file' for x in items),
                  'total_file_bytes': sum(x.get('bytes', 0) for x in items)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output)
        print(json.dumps({k: v for k, v in report.items() if k != 'entries'}))
    else:
        manifest = json.loads(args.manifest.read_text())
        expected = {item['path']: item for item in manifest['entries']}
        if len(expected) != len(manifest['entries']):
            raise ValueError('duplicate paths in manifest')
        observed = (inventory(args.root) if args.command == 'verify' else
                    archive_entries(args.archive, args.prefix))
        verify(expected, observed)


if __name__ == '__main__':
    main()
