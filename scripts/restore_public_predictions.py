"""Verify and restore public prediction Release shards with the Python standard library.

Run from the repository root after downloading both manifests and archive parts:
    python scripts/restore_public_predictions.py --assets /path/to/downloads
The default destination is the current working directory. No observations are included.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Parts:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.f = next(self.paths).open('rb')
        self.h = hashlib.sha256()
    def read(self, size):
        chunks = []
        while size:
            if self.f is None:
                break
            b = self.f.read(size)
            if b:
                chunks.append(b)
                size -= len(b)
            else:
                self.f.close()
                p = next(self.paths, None)
                if p is None:
                    self.f = None
                    break
                self.f = p.open('rb')
        b = b''.join(chunks)
        self.h.update(b)
        return b
    def close(self):
        if self.f is not None:
            self.f.close()


def restore(assets, destination, verify_only):
    manifests = sorted(assets.glob('MANIFEST_*.json'))
    if not manifests:
        raise FileNotFoundError('No MANIFEST_*.json files in asset directory')
    for mp in manifests:
        m = json.loads(mp.read_text(encoding='utf-8'))
        paths = []
        for p in m['archive']['parts']:
            path = assets / p['name']
            if path.stat().st_size != p['bytes'] or sha256(path) != p['sha256']:
                raise ValueError(f'Archive part mismatch: {path}')
            paths.append(path)
        expected = {f['path']: f for f in m['files']}
        seen = set()
        source = Parts(paths)
        with tarfile.open(fileobj=source, mode='r|gz') as archive:
            for entry in archive:
                name = entry.name
                safe = PurePosixPath(name)
                if not entry.isfile() or safe.is_absolute() or '..' in safe.parts or name not in expected or name in seen:
                    raise ValueError(f'Unexpected archive member: {name}')
                b = archive.extractfile(entry).read()
                record = expected[name]
                if len(b) != record['bytes'] or hashlib.sha256(b).hexdigest() != record['sha256']:
                    raise ValueError(f'File mismatch: {name}')
                if not verify_only:
                    target = (destination / name).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError(f'Unsafe target: {target}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b)
                seen.add(name)
        # Consume any compressed end padding before validating the combined hash.
        while source.read(8 * 1024 * 1024):
            pass
        source.close()
        if seen != set(expected) or source.h.hexdigest() != m['archive']['sha256']:
            raise ValueError(f'Incomplete or mismatched archive: {mp}')
        print(f"{m['group']}: verified {len(seen)} files ({m['prediction_count']} prediction NPZs)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', required=True, type=Path)
    ap.add_argument('--destination', type=Path, default=Path.cwd())
    ap.add_argument('--verify-only', action='store_true', help='Read and verify every file without extracting')
    args = ap.parse_args()
    restore(args.assets.resolve(), args.destination.resolve(), args.verify_only)
