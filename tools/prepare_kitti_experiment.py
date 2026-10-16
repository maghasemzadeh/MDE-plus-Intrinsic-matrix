#!/usr/bin/env python3
"""Build an isolated, drive-disjoint KITTI metric-depth experiment split.

The canonical KITTI tree and the official 652-image DA2 evaluation list are
read-only inputs.  This writes only ``splits/train.txt``, ``splits/val.txt``
under the requested experiment root.  RGB/depth paths are absolute, so the
training loader cannot silently substitute a frame from another drive.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


DRIVE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d+_sync")


def drive_name(path: Path) -> str:
    match = DRIVE_RE.search(str(path))
    if not match:
        raise ValueError(f"cannot infer KITTI drive from {path}")
    return match.group(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.kitti_root.resolve()
    out = args.experiment_root.resolve()
    eval_list = (root / "da2_val.txt").read_text().splitlines()
    eval_drives = {drive_name(Path(line.split()[0])) for line in eval_list if line.strip()}

    candidates: list[tuple[str, Path, Path]] = []
    depth_root = root / "train"
    for depth_path in sorted(depth_root.glob("*/proj_depth/groundtruth/image_02/*.png")):
        drive = drive_name(depth_path)
        if drive in eval_drives:
            continue
        frame = depth_path.stem
        date = drive[:10]
        image_path = root / "kitti_raw" / date / drive / "image_02" / "data" / f"{frame}.png"
        if image_path.exists():
            candidates.append((drive, image_path, depth_path))

    if not candidates:
        raise RuntimeError("no non-evaluation KITTI RGB/depth pairs found")

    by_drive: dict[str, list[tuple[str, Path, Path]]] = {}
    for row in candidates:
        by_drive.setdefault(row[0], []).append(row)

    rng = random.Random(args.seed)
    drives = sorted(by_drive)
    rng.shuffle(drives)
    # Reserve complete drives for validation.  This is deliberately disjoint
    # from both the training drives and all 652 official DA2 evaluation drives.
    val_drives = set(drives[: max(1, min(3, len(drives) // 10))])
    train_rows = [r for r in candidates if r[0] not in val_drives]
    val_rows = [r for r in candidates if r[0] in val_drives]

    if args.frame_stride > 1:
        train_rows = train_rows[:: args.frame_stride]
        val_rows = val_rows[:: args.frame_stride]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    train_rows = train_rows[: args.max_train]
    val_rows = val_rows[: args.max_val]

    if not train_rows or not val_rows:
        raise RuntimeError(f"empty split after filtering: train={len(train_rows)}, val={len(val_rows)}")

    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: list[tuple[str, Path, Path]]) -> None:
        text = "\n".join(f"{image} {depth}" for _, image, depth in rows) + "\n"
        (split_dir / name).write_text(text)

    write("train.txt", train_rows)
    write("val.txt", val_rows)
    metadata = [
        f"kitti_root={root}",
        f"eval_drives={len(eval_drives)}",
        f"train_drives={len({r[0] for r in train_rows})}",
        f"val_drives={len({r[0] for r in val_rows})}",
        f"train_samples={len(train_rows)}",
        f"val_samples={len(val_rows)}",
        f"frame_stride={args.frame_stride}",
        f"seed={args.seed}",
        "intrinsics=KITTI default calibration (721.5377,721.5377,609.5593,172.854) via loader",
        "eval_leakage_check=all official DA2 evaluation drives excluded",
        "val_drives=" + ",".join(sorted({r[0] for r in val_rows})),
    ]
    (out / "MANIFEST.txt").write_text("\n".join(metadata) + "\n")
    print("\n".join(metadata))


if __name__ == "__main__":
    main()
