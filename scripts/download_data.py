"""Download the Rasp & Lerch (2018) PPNN dataset from figshare and verify its MD5.

Source: Rasp, S. (2021): PPNN full data (feather format). figshare.
https://doi.org/10.6084/m9.figshare.13516301.v1  (license: CC BY 4.0)
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/25942496"
FILENAME = "data_RL18.feather"
MD5 = "18bb8d47b47692a2bda193572b0683ed"   # supplied_md5 from the figshare API
SIZE = 615496032


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()
    out = Path(args.out_dir) / FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == SIZE and md5sum(out) == MD5:
        print(f"{out} already present, MD5 OK")
        return 0
    tmp = out.with_suffix(".part")
    print(f"Downloading {URL} -> {out}")
    req = urllib.request.Request(URL, headers={"User-Agent": "ppnn-residual/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if done % (50 << 20) < (1 << 20):
                print(f"  {done / 1e6:.0f} MB", flush=True)
    got = md5sum(tmp)
    if got != MD5:
        print(f"MD5 mismatch: got {got}, expected {MD5}", file=sys.stderr)
        return 1
    tmp.rename(out)
    print(f"OK: {out} ({out.stat().st_size} bytes, MD5 {got})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
