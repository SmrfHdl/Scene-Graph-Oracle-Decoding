"""Stream-download Visual Genome zips → extract on the fly.

Why this exists:
    Reefknot needs Visual Genome images (~14 GB extracted). The CS Stanford
    mirror serves two zips totaling ~14 GB. We can't afford to store both
    zip AND extracted on a 23 GB-free disk, and `bsdtar -xf -` is unavailable
    without sudo. This script downloads via HTTP and writes each entry as it
    decompresses — peak disk usage = extracted size only.

Usage:
    # Install deps (user-level, no sudo)
    .venv/bin/python -m pip install stream-unzip httpx

    # Run
    .venv/bin/python scripts/stream_unzip_vg.py \\
        https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip \\
        data/reefknot/images

    .venv/bin/python scripts/stream_unzip_vg.py \\
        https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip \\
        data/reefknot/images
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from stream_unzip import stream_unzip


def _download_chunks(url: str, chunk_size: int = 1 << 16):
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        t0 = time.time()
        for chunk in r.iter_bytes(chunk_size=chunk_size):
            done += len(chunk)
            if total and (done % (50 << 20) < chunk_size or done == total):
                rate = done / max(time.time() - t0, 0.01) / (1 << 20)
                print(f"\r  downloaded {done/(1<<30):.2f} / {total/(1<<30):.2f} GB "
                      f"({100*done/total:.1f}%, {rate:.1f} MB/s)", end="", flush=True)
            yield chunk
        print()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    url, dest = sys.argv[1], Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)

    n_files = 0
    t0 = time.time()
    for raw_name, _file_size, chunks in stream_unzip(_download_chunks(url)):
        name = raw_name.decode("utf-8", errors="replace")
        # Flatten: strip any subdirectory (VG_100K/, VG_100K_2/) — we want all
        # images at the top level of dest/.
        flat = os.path.basename(name.rstrip("/"))
        if name.endswith("/") or not flat:
            # directory entry; skip
            for _ in chunks:
                pass
            continue
        out_path = dest / flat
        with open(out_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
        n_files += 1
        if n_files % 1000 == 0:
            print(f"  extracted {n_files} files ({time.time()-t0:.0f}s)")
    print(f"Done: {n_files} files in {time.time()-t0:.0f}s → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
