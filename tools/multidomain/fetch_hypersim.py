#!/usr/bin/env python3
"""Fetch a Hypersim subset for multi-domain metric training.

Only scenes rendered with the standard 60-degree pinhole camera are used
(M_cam_from_uv diagonal in contrib/mikeroberts3000/metadata_camera_parameters.csv;
tilt-shift scenes would need a non-standard K). Frames follow Hypersim's
official scene split (evermotion_dataset/analysis/metadata_images_split_scene_v1.csv),
keeping only images in the public release without an exclusion reason. Each
scene zip is read with HTTP range requests (remotezip), so only the needed
members are transferred:

    images/scene_<cam>_final_preview/frame.<id>.tonemap.jpg  -> image.jpg (bytes as-is)
    images/scene_<cam>_geometry_hdf5/frame.<id>.depth_meters.hdf5
        Euclidean distance to the camera centre -> planar depth
        Z = d / sqrt(1 + (u M00)^2 + (v M11)^2), u, v in [-1, 1] at pixel
        centres (the same conversion DA2 uses), stored as uint16 mm PNG,
        0 = invalid.

Output tree: <out>/<split>/<scene>_<cam>_<id>/{image.jpg, depth.png}
"""
import argparse
import csv
import io
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
from remotezip import RemoteZip

BASE = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes'
OFF_DIAGONAL = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def standard_scenes(cam_csv):
    """scene -> (M00, M11) for scenes with a plain 1024x768 pinhole camera."""
    out = {}
    for r in csv.DictReader(open(cam_csv)):
        M = [[float(r[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)]
        plain = all(abs(M[i][j]) < 1e-6 for i, j in OFF_DIAGONAL) and abs(M[2][2] + 1) < 1e-6
        if plain and float(r['settings_output_img_width']) == 1024 \
                and float(r['settings_output_img_height']) == 768:
            out[r['scene_name']] = (M[0][0], M[1][1])
    return out


def select(split_csv, scenes, split, per_scene, seed):
    """scene -> sorted [(camera, frame_id)], at most per_scene random frames each."""
    by_scene = defaultdict(list)
    for r in csv.DictReader(open(split_csv)):
        if (r['split_partition_name'] == split and r['scene_name'] in scenes
                and r['included_in_public_release'] == 'True' and not r['exclude_reason']):
            by_scene[r['scene_name']].append((r['camera_name'], int(r['frame_id'])))
    rng = random.Random(seed)
    picked = {}
    for scene in sorted(by_scene):
        frames = sorted(by_scene[scene])
        rng.shuffle(frames)
        picked[scene] = sorted(frames[:per_scene])
    return picked


def ray_norm(m00, m11, h=768, w=1024):
    u = ((np.arange(w) + 0.5) / w * 2 - 1) * m00
    v = (1 - (np.arange(h) + 0.5) / h * 2) * m11
    uu, vv = np.meshgrid(u, v)
    return np.sqrt(1.0 + uu ** 2 + vv ** 2).astype(np.float32)


def fetch_scene(scene, frames, split, out, m):
    def dst(cam, fid):
        return out / split / f'{scene}_{cam}_{fid:04d}'
    todo = [(c, f) for c, f in frames if not (dst(c, f) / 'depth.png').exists()]
    if not todo:
        return scene, 0, 0
    norm = ray_norm(*m)
    n = missing = 0
    with RemoteZip(f'{BASE}/{scene}.zip') as z:
        for cam, fid in todo:
            try:
                jpg = z.read(f'{scene}/images/scene_{cam}_final_preview/frame.{fid:04d}.tonemap.jpg')
                h5 = z.read(f'{scene}/images/scene_{cam}_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5')
            except KeyError:
                missing += 1
                continue
            dist = h5py.File(io.BytesIO(h5), 'r')['dataset'][:].astype(np.float32)
            depth = dist / norm
            ok = np.isfinite(depth) & (depth > 0)
            mm = np.zeros(depth.shape, np.uint16)
            mm[ok] = np.clip(np.round(depth[ok] * 1000.0), 1, 65535).astype(np.uint16)
            d = dst(cam, fid)
            d.mkdir(parents=True, exist_ok=True)
            (d / 'image.jpg').write_bytes(jpg)
            if not cv2.imwrite(str(d / 'depth.tmp.png'), mm):
                raise OSError(f'cannot write {d}/depth.tmp.png')
            (d / 'depth.tmp.png').rename(d / 'depth.png')  # written last: marks the frame complete
            n += 1
    return scene, n, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--meta', type=Path, required=True,
                   help='directory with metadata_camera_parameters.csv and '
                        'metadata_images_split_scene_v1.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-scene', type=int, default=40, help='train frames per scene')
    p.add_argument('--val-per-scene', type=int, default=3)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    scenes = standard_scenes(a.meta / 'metadata_camera_parameters.csv')
    split_csv = a.meta / 'metadata_images_split_scene_v1.csv'
    jobs = []
    for split, k in (('train', a.per_scene), ('val', a.val_per_scene)):
        picked = select(split_csv, scenes, split, k, a.seed)
        print(f"{split}: {len(picked)} scenes, {sum(map(len, picked.values()))} frames", flush=True)
        jobs += [(s, f, split) for s, f in picked.items()]

    total = sum(len(f) for _, f, _ in jobs)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_scene, s, f, split, a.out, scenes[s]): (s, len(f))
                for s, f, split in jobs}
        for fut in as_completed(futs):
            scene, m = futs[fut]
            try:
                _, n, missing = fut.result()
            except Exception as e:  # network / zip errors: rerun resumes from depth.png markers
                print(f"FAILED {scene}: {e}", file=sys.stderr, flush=True)
                continue
            done += m
            print(f"{scene}: +{n}{f' ({missing} missing)' if missing else ''} ({done}/{total})", flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
