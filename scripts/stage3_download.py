"""Stage 3: download the EUPPBench station-data subset used in Stage 3 and verify it.

The data are the Zarr stores of the EUPPBench v1.0 station dataset (Zenodo 10.5281/zenodo.7708362,
EUPPBench-stations.zip). The zip is stored uncompressed (ZIP_STORED) and every member carries a CRC32
in the central directory. This script
  1. reads the central directory of the Zenodo zip via HTTP range requests (no full download),
  2. downloads the needed members (selected variables + all metadata/coordinate arrays) from the
     official climetlab/ECMWF object store (same files, much faster), falling back to extracting the
     member from the Zenodo zip by range request,
  3. verifies size and CRC32 of every file against the Zenodo central directory.
Usage: python scripts/stage3_download.py --out /path/to/euppbench [--workers 16]
"""
import argparse
import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ZENODO_ZIP = "https://zenodo.org/api/records/7708362/files/EUPPBench-stations.zip/content"
ZENODO_MD5 = "e409457279b3494d18f2dfb41f3f449b"
ZENODO_SIZE = 18324361137
OBJ = "https://object-store.os-api.cci1.ecmwf.int/eumetnet-postprocessing-benchmark-1st-phase-training-dataset/data/stations_data/"
COUNTRIES = ["austria", "belgium", "france", "germany", "netherlands"]
SURFACE = ["t2m", "u10", "v10", "tcc", "sd", "stl1", "swvl1", "tcwv", "cape"]
POSTPROC = ["mx2t6", "mn2t6", "sshf6", "slhf6", "ssr6", "str6"]
P850 = ["t"]
OBS = ["t2m"]
DATA_VARS_ALL = {"cape", "cin", "sd", "stl1", "swvl1", "t2m", "tcc", "tcw", "tcwv", "u10", "u100", "v10", "v100", "vis",
                 "cp6", "mn2t6", "mx2t6", "p10fg6", "slhf6", "sshf6", "ssr6", "ssrd6", "str6", "strd6", "tp6", "t"}


def stores():
    out = {}
    for c in COUNTRIES:
        for kind in ("reforecasts", "forecasts"):
            out[f"stations_ensemble_{kind}_surface_{c}.zarr"] = SURFACE
            out[f"stations_ensemble_{kind}_surface_postprocessed_{c}.zarr"] = POSTPROC
            out[f"stations_ensemble_{kind}_pressure_850_{c}.zarr"] = P850
            out[f"stations_{kind}_observations_surface_{c}.zarr"] = OBS
    return out


def listing(cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    from remotezip import RemoteZip
    with RemoteZip(ZENODO_ZIP) as z:
        rows = {i.filename: [i.file_size, i.CRC, i.compress_type] for i in z.infolist()}
    cache.write_text(json.dumps(rows))
    return rows


def wanted(rows):
    st = stores()
    sel = []
    for name, (size, crc, ct) in rows.items():
        if name.endswith("/"):
            continue
        parts = name.split("/")
        if parts[0] not in st:
            continue
        if len(parts) == 2:  # store-level metadata
            sel.append(name)
        elif parts[1] in st[parts[0]] or parts[1] not in DATA_VARS_ALL:  # selected var or coordinate array
            sel.append(name)
    return sorted(sel)


def fetch(name, size, crc, out: Path, session):
    dst = out / name
    if dst.exists() and dst.stat().st_size == size and (zlib.crc32(dst.read_bytes()) & 0xFFFFFFFF) == crc:
        return name, "cached", size
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            r = session.get(OBJ + name, timeout=120)
            if r.status_code == 200:
                b = r.content
                if len(b) == size and (zlib.crc32(b) & 0xFFFFFFFF) == crc:
                    dst.write_bytes(b)
                    return name, "object-store", size
        except requests.RequestException:
            pass
    # fallback: extract the member from the Zenodo zip
    from remotezip import RemoteZip
    with RemoteZip(ZENODO_ZIP) as z:
        b = z.read(name)
    if len(b) != size or (zlib.crc32(b) & 0xFFFFFFFF) != crc:
        raise RuntimeError(f"verification failed for {name}")
    dst.write_bytes(b)
    return name, "zenodo-zip", size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = listing(out / "zenodo_zip_central_directory.json")
    assert all(v[2] == 0 for v in rows.values()), "zip members expected to be STORED"
    sel = wanted(rows)
    tot = sum(rows[n][0] for n in sel)
    print(f"{len(sel)} files, {tot/1e9:.2f} GB to fetch/verify", flush=True)
    src_count, done = {}, 0
    with ThreadPoolExecutor(a.workers) as ex:
        sess = [requests.Session() for _ in range(a.workers)]
        futs = [ex.submit(fetch, n, rows[n][0], rows[n][1], out, sess[i % a.workers]) for i, n in enumerate(sel)]
        for i, f in enumerate(as_completed(futs)):
            n, src, size = f.result()
            src_count[src] = src_count.get(src, 0) + 1
            done += size
            if i % 500 == 0:
                print(f"{i+1}/{len(sel)} files, {done/1e9:.2f} GB", src_count, flush=True)
    manifest = {"zenodo_record": "10.5281/zenodo.7708362", "zenodo_zip_md5": ZENODO_MD5, "zenodo_zip_size": ZENODO_SIZE,
                "n_files": len(sel), "bytes": tot, "sources": src_count,
                "verification": "every file: size and CRC32 equal to the Zenodo zip central directory entry",
                "stores": stores()}
    (out / "stage3_download_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", manifest["sources"], flush=True)


if __name__ == "__main__":
    main()
