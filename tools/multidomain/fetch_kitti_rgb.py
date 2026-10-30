#!/usr/bin/env python3
"""Fetch only the KITTI raw image_02 frames listed by select_kitti_frames.py.

Runs on a machine that can reach the KITTI S3 bucket. Each drive's sync zip
is read with HTTP range requests (remotezip), so only the listed PNG members
are transferred. Frames are re-encoded as JPEG (quality 95) to keep the
upload to the GPU server small; per-date calib_cam_to_cam.txt files are
fetched alongside so the loader can use the true rectified K per date.

Output tree (mirrors KITTI raw):
    <out>/<date>/calib_cam_to_cam.txt
    <out>/<date>/<drive>/image_02/data/<frame>.jpg
"""
import argparse
import io
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'


def fetch_drive(date, drive, frames, out, quality):
    url = f"{BASE}/{drive[:-5]}/{drive}.zip"
    dst = out / date / drive / 'image_02' / 'data'
    dst.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if not (dst / f"{f}.jpg").exists()]
    if not todo:
        return drive, 0, len(frames)
    n = 0
    with RemoteZip(url) as z:
        for f in todo:
            member = f"{date}/{drive}/image_02/data/{f}.png"
            img = Image.open(io.BytesIO(z.read(member))).convert('RGB')
            tmp = dst / f"{f}.jpg.tmp"
            img.save(tmp, format='JPEG', quality=quality)
            tmp.rename(dst / f"{f}.jpg")
            n += 1
    return drive, n, len(frames)


def fetch_calib(date, out):
    dst = out / date / 'calib_cam_to_cam.txt'
    if dst.exists():
        return
    with RemoteZip(f"{BASE}/{date}_calib.zip") as z:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(f"{date}/calib_cam_to_cam.txt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--quality', type=int, default=95)
    a = p.parse_args()

    by_drive = defaultdict(list)
    dates = set()
    for line in a.frames.read_text().splitlines():
        if not line.strip():
            continue
        _, date, drive, frame, _ = line.split()
        by_drive[(date, drive)].append(frame)
        dates.add(date)

    for date in sorted(dates):
        fetch_calib(date, a.out)
    print(f"calib fetched for {sorted(dates)}", flush=True)

    total = sum(len(v) for v in by_drive.values())
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_drive, d, dr, fr, a.out, a.quality): dr
                for (d, dr), fr in sorted(by_drive.items())}
        for fut in as_completed(futs):
            try:
                drive, n, m = fut.result()
            except (zipfile.BadZipFile, KeyError, OSError) as e:
                print(f"FAILED {futs[fut]}: {e}", file=sys.stderr, flush=True)
                continue
            done += m
            print(f"{drive}: +{n} ({done}/{total})", flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
