#!/usr/bin/env python3
"""Pick drive-disjoint KITTI training frames that have annotated GT depth.

Excludes every drive that contributes to the 652-image DA2/eigen evaluation
list, holds out two further drives for in-training validation, and keeps
every `stride`-th annotated image_02 frame of the remaining drives. Output is
one line per frame: "<split> <date> <drive> <frame> <gt_path>".
"""
import argparse
import random
import re
from pathlib import Path

DRIVE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})_drive_\d+_sync")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kitti-root', type=Path, required=True)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--n-val-drives', type=int, default=2)
    p.add_argument('--val-frames-per-drive', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    eval_drives = set()
    for line in (a.kitti_root / 'da2_val.txt').read_text().splitlines():
        m = DRIVE_RE.search(line)
        if m:
            eval_drives.add(m.group(0))

    gt = {}
    for sub in ('train', 'val'):
        for f in sorted((a.kitti_root / sub).glob('*/proj_depth/groundtruth/image_02/*.png')):
            drive = f.parts[-5]
            gt.setdefault(drive, []).append(f)
    drives = sorted(d for d in gt if d not in eval_drives)
    rng = random.Random(a.seed)
    shuffled = drives[:]
    rng.shuffle(shuffled)
    val_drives = set(shuffled[:a.n_val_drives])

    rows = []
    for d in drives:
        frames = gt[d]
        date = DRIVE_RE.search(d).group(1)
        if d in val_drives:
            step = max(1, len(frames) // a.val_frames_per_drive)
            picked, split = frames[::step][:a.val_frames_per_drive], 'val'
        else:
            picked, split = frames[::a.stride], 'train'
        rows += [f"{split} {date} {d} {f.stem} {f}" for f in picked]

    a.out.write_text('\n'.join(rows) + '\n')
    n_train = sum(r.startswith('train') for r in rows)
    print(f"eval drives excluded: {len(eval_drives)}; candidate drives: {len(drives)}; "
          f"val drives: {sorted(val_drives)}")
    print(f"train frames: {n_train}; val frames: {len(rows) - n_train}")


if __name__ == '__main__':
    main()
