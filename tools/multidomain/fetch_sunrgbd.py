#!/usr/bin/env python3
"""Fetch a per-sensor random subset of SUN RGB-D for zero-shot evaluation.

Reads SUNRGBD.zip by HTTP range requests (remotezip): for each sampled frame
folder only image/*.jpg, the raw depth/*.png and intrinsics.txt are pulled.
kv1/NYUdata is skipped (it is the NYU labeled set, overlapping our NYU
training images). Sensor groups:
  kv2        kv2/kinect2data, kv2/align_kv2           (Kinect v2)
  realsense  realsense/{lg,sa,sh,shr}                 (Intel RealSense)
  xtion      xtion/{sun3ddata,xtion_align_data}       (Asus Xtion)
  kv1        kv1/b3dodata                             (Kinect v1, B3DO)
Output: <out>/<sensor>/<folder>/{image.jpg, depth.png, intrinsics.txt}
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = 'https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
GROUPS = {
    'kv2': ('kv2/kinect2data/', 'kv2/align_kv2/'),
    'realsense': ('realsense/lg/', 'realsense/sa/', 'realsense/sh/', 'realsense/shr/'),
    'xtion': ('xtion/sun3ddata/', 'xtion/xtion_align_data/'),
    'kv1': ('kv1/b3dodata/',),
}


def frame_folders(names):
    """folder -> {'image': member, 'depth': member, 'K': member} for complete frames."""
    folders = defaultdict(dict)
    for n in names:
        parts = n.split('/')
        if len(parts) < 4 or n.endswith('/'):
            continue
        if parts[-2] == 'image' and n.endswith('.jpg'):
            folders['/'.join(parts[:-2])]['image'] = n
        elif parts[-2] == 'depth' and n.endswith('.png'):
            folders['/'.join(parts[:-2])]['depth'] = n
        elif parts[-1] == 'intrinsics.txt':
            folders['/'.join(parts[:-1])]['K'] = n
    return {f: v for f, v in folders.items() if len(v) == 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--namelist', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-sensor', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()

    folders = frame_folders(json.loads(a.namelist.read_text()))
    rng = random.Random(a.seed)
    jobs = []
    for sensor, prefixes in GROUPS.items():
        cands = sorted(f for f in folders if any(f.startswith('SUNRGBD/' + q) for q in prefixes))
        picked = rng.sample(cands, min(a.per_sensor, len(cands)))
        print(f"{sensor}: {len(cands)} frames available, {len(picked)} sampled", flush=True)
        jobs += [(sensor, f) for f in picked]

    def fetch_chunk(chunk):
        # one RemoteZip per worker: opening it re-reads the ~40 MB central directory
        with RemoteZip(URL) as z:
            for sensor, folder in chunk:
                dst = a.out / sensor / folder.rstrip('/').split('/')[-1]
                if (dst / 'intrinsics.txt').exists():
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                # intrinsics.txt last: its presence marks a complete folder
                for key, fn in (('image', 'image.jpg'), ('depth', 'depth.png'), ('K', 'intrinsics.txt')):
                    (dst / fn).write_bytes(z.read(folders[folder][key]))

    chunks = [jobs[i::a.workers] for i in range(a.workers)]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(fetch_chunk, chunks))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
