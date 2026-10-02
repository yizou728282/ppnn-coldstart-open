"""Validate preserved release assets and publish only predictions plus row indexes."""
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

# Optional second argument: a working directory containing stage2-4/ and stage5/ source assets.
ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parent
OUT = ROOT / 'public'
OUT.mkdir(exist_ok=True)
EXPECTED = {
    'original_preds_stage2-4.tar.gz.part00': '32c43eaab5f3217e0badf762d505c61a5e4a60b7166979cdd8df5988533e4e66',
    'original_preds_stage2-4.tar.gz.part01': 'a7fba63b14e473eeebf1935f4df40a6d617dc8f59ebef2ea5df9727bcdc287ed',
    'stage5_preds.tar.part00': 'd772245db92bc028e2be98e0ffa439493ffa5feb78b45c26387b79d1218aa93e',
    'stage5_preds.tar.part01': 'f3abd6d791c65ffa1ee8e840407b4232337d5476138e9ad9e7217a187229d31b',
}

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

class Parts:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.f = next(self.paths).open('rb')
    def read(self, size=-1):
        assert size >= 0
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
        return b''.join(chunks)
    def close(self):
        if self.f is not None:
            self.f.close()

class Shards:
    def __init__(self, base):
        self.base = base
        self.limit = 1_000_000_000
        self.pos = 0
        self.part_pos = 0
        self.index = 0
        self.h = hashlib.sha256()
        self.files = []
        self.open_part()
    def open_part(self):
        p = self.base.with_name(f'{self.base.name}.part{self.index:02d}')
        self.f = p.open('wb')
        self.files.append(p)
        self.part_pos = 0
    def write(self, b):
        original_len = len(b)
        self.h.update(b)
        self.pos += original_len
        while b:
            if self.part_pos == self.limit:
                self.f.close()
                self.index += 1
                self.open_part()
            chunk = b[:self.limit - self.part_pos]
            self.f.write(chunk)
            self.part_pos += len(chunk)
            b = b[len(chunk):]
        return original_len
    def tell(self):
        return self.pos
    def flush(self):
        self.f.flush()
    def close(self):
        self.f.close()

def add(tf, name, b):
    t = tarfile.TarInfo(name)
    t.size = len(b)
    t.mode = 0o644
    t.mtime = 0
    tf.addfile(t, io.BytesIO(b))

def run(group):
    folder = ROOT / group
    parts = sorted(folder.glob('*.part*'))
    assert len(parts) == 2
    for p in parts:
        assert digest_file(p) == EXPECTED[p.name], f'Source checksum mismatch: {p}'
        print('Verified source', p.name, flush=True)
    manifest = {'group': group, 'source_assets': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': EXPECTED[p.name]} for p in parts], 'files': [], 'excluded': [], 'schemas': {}}
    archive = OUT / f'predictions_{group}.tar.gz'
    src = Parts(parts)
    dst = Shards(archive)
    schemas = Counter()
    count = 0
    with tarfile.open(fileobj=src, mode='r|gz' if group == 'stage2-4' else 'r|') as ti, tarfile.open(fileobj=dst, mode='w|gz', compresslevel=1) as to:
        for m in ti:
            if not m.isfile():
                continue
            name = m.name.lstrip('./')
            assert not name.startswith('/') and '..' not in Path(name).parts
            is_meta = name.endswith('/test_meta.npz')
            is_pred = '/preds/' in name and name.endswith('.npz') and not is_meta
            is_fold = name.endswith(('folds_random.csv', 'folds_spatial.csv'))
            if not (is_meta or is_pred or is_fold):
                manifest['excluded'].append({'path': name, 'reason': 'Outside prediction/index scope'})
                continue
            b = ti.extractfile(m).read()
            original_hash = hashlib.sha256(b).hexdigest()
            published_name = name
            removed = []
            if is_meta:
                # Load identifiers only and eliminate object dtypes.
                with np.load(io.BytesIO(b), allow_pickle=True) as z:
                    fields = {}
                    for key in ('station', 'date', 'lead', 'init', 'unseen'):
                        if key in z.files:
                            a = z[key]
                            fields[key] = a.astype(str) if a.dtype.kind == 'O' else a
                    removed = sorted(set(z.files) - set(fields))
                assert fields and 'station' in fields
                bio = io.BytesIO()
                np.savez_compressed(bio, **fields)
                b = bio.getvalue()
                published_name = name.replace('test_meta.npz', 'test_index.npz')
            elif is_pred:
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    keys = [n.removesuffix('.npy') for n in z.namelist()]
                    allowed = {'mu', 'sigma', 'mu11', 'sigma11', 'seeds', 'sum_mu', 'sum_sigma', 'station', 'rows'}
                    unknown = set(keys) - allowed - {'y'}
                    assert not unknown, (name, unknown)
                    assert {'mu', 'sigma'} <= set(keys) or {'sum_mu', 'sum_sigma', 'seeds'} <= set(keys), (name, keys)
                    if 'y' in keys:
                        # Preserve every predictive .npy payload byte-for-byte; remove observations.
                        bio = io.BytesIO()
                        with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_STORED) as clean:
                            for n in z.namelist():
                                if n != 'y.npy':
                                    clean.writestr(n, z.read(n))
                        b = bio.getvalue()
                        removed = ['y']
                    schemas[','.join(sorted(set(keys) - set(removed)))] += 1
                count += 1
            add(to, published_name, b)
            manifest['files'].append({'path': published_name, 'kind': 'prediction' if is_pred else 'index' if is_meta else 'fold', 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'source_sha256': original_hash, 'removed_fields': removed, 'seed_specific': bool(is_pred and re.search(r'_s\d+\.npz$', name))})
            if count and count % 200 == 0 and is_pred:
                print(group, count, 'predictions packaged', flush=True)
    src.close()
    dst.close()
    manifest['prediction_count'] = count
    manifest['schemas'] = dict(schemas)
    manifest['archive'] = {'name': archive.name, 'bytes': dst.pos, 'sha256': dst.h.hexdigest(), 'parts': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in dst.files]}
    (OUT / f'MANIFEST_{group}.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'group': group, 'predictions': count, 'schemas': dict(schemas), 'archive': manifest['archive']}), flush=True)

if __name__ == '__main__':
    run(sys.argv[1])
