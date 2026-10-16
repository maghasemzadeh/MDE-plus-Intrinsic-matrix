#!/usr/bin/env python3
"""Create scene-disjoint VKITTI2 train/validation lists for a new run."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


SCENE_RE = re.compile(r"(Scene\d+)")


def scene(line: str) -> str:
    match = SCENE_RE.search(line)
    if not match:
        raise ValueError(f"cannot infer VKITTI scene from {line[:160]}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vkitti-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=36000)
    parser.add_argument("--max-val", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.vkitti_root.resolve()
    lines = [line.strip() for line in (root / "splits/train.txt").read_text().splitlines() if line.strip()]
    groups: dict[str, list[str]] = {}
    for line in lines:
        groups.setdefault(scene(line), []).append(line)
    names = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(names)
    val_scenes = set(names[: max(1, min(3, len(names) // 10))])
    train = [line for key, rows in groups.items() if key not in val_scenes for line in rows]
    val = [line for key, rows in groups.items() if key in val_scenes for line in rows]
    rng.shuffle(train)
    rng.shuffle(val)
    train, val = train[: args.max_train], val[: args.max_val]
    if not train or not val:
        raise RuntimeError(f"empty split: train={len(train)} val={len(val)}")

    out = args.experiment_root.resolve()
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits/train.txt").write_text("\n".join(train) + "\n")
    (out / "splits/val.txt").write_text("\n".join(val) + "\n")
    metadata = [
        f"vkitti_root={root}",
        f"train_scenes={','.join(sorted({scene(x) for x in train}))}",
        f"val_scenes={','.join(sorted({scene(x) for x in val}))}",
        f"train_samples={len(train)}",
        f"val_samples={len(val)}",
        f"seed={args.seed}",
        "eval_protocol=real KITTI official 652-image DA2 list, held out from training",
    ]
    (out / "MANIFEST.txt").write_text("\n".join(metadata) + "\n")
    print("\n".join(metadata))


if __name__ == "__main__":
    main()
