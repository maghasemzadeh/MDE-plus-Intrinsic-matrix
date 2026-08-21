#!/usr/bin/env python3
"""Decide whether thesis section 4.4's improvement column survives a correct
depth/disparity alignment.

Section 4.4's tables were produced by the January 2026 pipeline, which
median-scaled predictions in *depth* space and never inverted them.  The
released Depth Anything V2 basic model outputs inverse depth (disparity), while
the fine-tuned "revised" checkpoints used in section 4.4 predate the
disparity-target fix and output *direct* depth.  If that is what happened, the
baseline column was scored in the wrong units and the reported 55-74% gain is
an artefact rather than a camera-intrinsics effect.

This script settles it by evaluating both checkpoints on the same images, with
the same valid-pixel mask, under two alignments that differ in exactly one way:

  historical  median depth scaling, no inversion  (reproduces the thesis)
  corrected   affine scale+shift, inverting when the model says it outputs
              inverse depth              (src.metrics.align_non_metric_predictions)

Run from the repository root on the machine that holds the checkpoints and the
raw datasets::

    source venv/bin/activate
    python tools/audit_section_4_4_units.py --max-items 200

Use --max-items for a fast verdict; omit it to evaluate every image.  The
sample size does not change the conclusion, because the effect under test is a
units mismatch that applies to every image identically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets import (  # noqa: E402
    CityscapesDataset, DatasetConfig, DIODEDataset, DrivingStereoDataset,
    NYUDataset, VKITTIDataset,
)
from models import DepthAnythingV2RevisedWrapper, DepthAnythingV2Wrapper  # noqa: E402
from src.metrics import align_non_metric_predictions  # noqa: E402

INPUT_SIZE = 518

VITL_BASE = "models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth"
VITS_BASE = "models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth"
MULTI_PROPOSED = (
    "models/raw_models/DepthAnythingV2-revised/checkpoints/revised/"
    "best-basic-vitl-multi-database.pth"
)
DRIVING_PROPOSED = (
    "models/raw_models/DepthAnythingV2-revised/checkpoints/revised/"
    "basic_vits_vkitti_intrinsics_lr5e-6_bs4_20251229_015011/best.pth"
)

# The four rows of thesis table 4.2 (tab:da2_revised_comparison), with the
# checkpoints recorded in tools/validate_section_4_4_2.py.  Note that the
# DrivingStereo row is a ViT-S pair while the other three are ViT-L.
RUNS = (
    # thesis name, dataset class, encoder, baseline ckpt, proposed ckpt, thesis base/proposed
    ("NYUv2",         NYUDataset,           "vitl", VITL_BASE, MULTI_PROPOSED,   1.284, 0.329),
    ("DIODE",         DIODEDataset,         "vitl", VITL_BASE, MULTI_PROPOSED,   0.833, 0.377),
    ("CityScapes",    CityscapesDataset,    "vitl", VITL_BASE, MULTI_PROPOSED,   1.835, 0.662),
    ("DrivingStereo", DrivingStereoDataset, "vits", VITS_BASE, DRIVING_PROPOSED, 1.399, 0.414),
)

EXTRA = {"VKITTI": (VKITTIDataset, "vitl", VITL_BASE, MULTI_PROPOSED, None, None)}


def silog(diff_log: np.ndarray) -> float:
    """The project's SILog: sqrt(mean(d^2) - 0.5*mean(d)^2)."""
    return float(np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2))


def historical_silog(pred_valid: np.ndarray, gt_valid: np.ndarray) -> float:
    """January 2026 protocol, byte-for-byte as in validate_section_4_4_1.py."""
    first_scale = np.median(gt_valid) / np.median(pred_valid)
    pred_scaled = (pred_valid * first_scale).astype(np.float32)
    second_scale = np.median(gt_valid) / np.median(pred_scaled)
    pred_scaled = pred_scaled * second_scale
    return silog(np.log(pred_scaled) - np.log(gt_valid))


def corrected_silog(pred_valid, gt_valid, outputs_inverse_depth: bool) -> float:
    """Same pixels, same metric — only the alignment is fixed."""
    aligned = align_non_metric_predictions(
        pred_valid.astype(np.float64), gt_valid.astype(np.float64),
        outputs_inverse_depth=outputs_inverse_depth,
    )
    aligned = np.maximum(aligned, 1e-8)
    return silog(np.log(aligned) - np.log(gt_valid))


def valid_pair(pred, gt, max_depth):
    """The historical valid mask: finite, strictly positive, optional depth cap."""
    pred = np.squeeze(np.asarray(pred, dtype=np.float32))
    gt = np.squeeze(np.asarray(gt, dtype=np.float32))
    if pred.shape != gt.shape:
        import cv2
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if max_depth:
        mask &= gt <= max_depth
    if int(mask.sum()) < 10:
        return None, None
    return pred[mask], gt[mask]


def build_model(kind: str, encoder: str, checkpoint: str, device):
    config = {"model_type": "basic", "encoder": encoder,
              "checkpoint_path": checkpoint, "device": device}
    model = (DepthAnythingV2Wrapper if kind == "baseline"
             else DepthAnythingV2RevisedWrapper)(config)
    model.load_model()
    return model


def evaluate(model, dataset, items, max_depth, label):
    from tqdm import tqdm
    inverse = bool(model.outputs_inverse_depth())
    hist, corr, skipped = [], [], 0
    for item in tqdm(items, desc=label, unit="img"):
        try:
            image = dataset.load_image(item)
            gt = dataset.load_gt_depth(item.gt_path, item)
            K = dataset.load_intrinsic(item) if dataset.has_intrinsics() else None
            pred = model.infer_image(image, input_size=INPUT_SIZE, intrinsics=K)
            p, g = valid_pair(pred, gt, max_depth)
            if p is None:
                skipped += 1
                continue
            hist.append(historical_silog(p, g))
            corr.append(corrected_silog(p, g, inverse))
        except Exception:
            skipped += 1
    return {
        "outputs_inverse_depth": inverse,
        "n": len(hist),
        "n_skipped": skipped,
        "silog_historical": float(np.mean(hist)) if hist else float("nan"),
        "silog_corrected": float(np.mean(corr)) if corr else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=[r[0] for r in RUNS],
                    help="subset of NYUv2 DIODE CityScapes DrivingStereo VKITTI")
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap images per dataset (recommended: 200 for a fast verdict)")
    ap.add_argument("--max-depth", type=float, default=None,
                    help="optional GT depth cap in metres (section 4.4 used 10 m for DIODE)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/section_4_4_units_audit.json")
    args = ap.parse_args()

    table = {r[0]: r[1:] for r in RUNS}
    table.update(EXTRA)

    results = {}
    for name in args.datasets:
        if name not in table:
            print(f"unknown dataset {name!r}; choose from {sorted(table)}", file=sys.stderr)
            return 2
        cls, encoder, base_ckpt, prop_ckpt, thesis_base, thesis_prop = table[name]

        for path in (base_ckpt, prop_ckpt):
            if not os.path.isfile(os.path.join(PROJECT_ROOT, path)):
                print(f"FAIL: missing checkpoint {path}", file=sys.stderr)
                return 1

        max_depth = args.max_depth
        if name == "DIODE" and max_depth is None:
            max_depth = 10.0  # the cap the January DIODE run used

        dataset = cls(DatasetConfig(max_items=args.max_items, force_evaluate=True))
        items = dataset.find_items()
        if args.max_items:
            items = items[:args.max_items]
        print(f"\n{'=' * 74}\n{name}: {len(items)} images, encoder {encoder}, "
              f"GT cap {max_depth or 'none'}\n{'=' * 74}")

        row = {"encoder": encoder, "n_items": len(items), "max_depth": max_depth,
               "thesis_baseline": thesis_base, "thesis_proposed": thesis_prop}
        for kind, ckpt in (("baseline", base_ckpt), ("proposed", prop_ckpt)):
            model = build_model(kind, encoder, os.path.join(PROJECT_ROOT, ckpt), args.device)
            row[kind] = evaluate(model, dataset, items, max_depth, f"{name}/{kind}")
            row[kind]["checkpoint"] = ckpt
            print(f"  {kind:<9} outputs_inverse_depth={row[kind]['outputs_inverse_depth']}  "
                  f"SILog historical={row[kind]['silog_historical']:.4f}  "
                  f"corrected={row[kind]['silog_corrected']:.4f}")
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        results[name] = row

    print("\n" + "=" * 96)
    print("SECTION 4.4 IMPROVEMENT COLUMN, BEFORE AND AFTER FIXING THE ALIGNMENT")
    print("=" * 96)
    print(f"{'dataset':<15}{'base':>9}{'prop':>9}{'improv':>9}   |"
          f"{'base':>9}{'prop':>9}{'improv':>9}   {'thesis':>9}")
    print(f"{'':<15}{'--- corrected alignment ---':^27}   |"
          f"{'--- historical (thesis) ---':^27}")
    print("-" * 96)
    for name, row in results.items():
        hb, hp = row["baseline"]["silog_historical"], row["proposed"]["silog_historical"]
        cb, cp = row["baseline"]["silog_corrected"], row["proposed"]["silog_corrected"]
        h_imp = (hb - hp) / hb * 100.0 if hb else float("nan")
        c_imp = (cb - cp) / cb * 100.0 if cb else float("nan")
        thesis = row["thesis_baseline"]
        t_imp = ((thesis - row["thesis_proposed"]) / thesis * 100.0) if thesis else float("nan")
        row["improvement_corrected_pct"] = c_imp
        row["improvement_historical_pct"] = h_imp
        print(f"{name:<15}{cb:>9.4f}{cp:>9.4f}{c_imp:>8.1f}%   |"
              f"{hb:>9.4f}{hp:>9.4f}{h_imp:>8.1f}%   {t_imp:>8.1f}%")
    print("=" * 96)
    print(
        "How to read this:\n"
        "  * 'historical' should land near the thesis column — that confirms the run\n"
        "    reproduces section 4.4 and the comparison below is like-for-like.\n"
        "  * If the baseline reports outputs_inverse_depth=True while the proposed\n"
        "    model reports False, the two columns of table 4.2 were scored in\n"
        "    different units.\n"
        "  * If the corrected improvement collapses toward zero (or turns negative)\n"
        "    while the historical one stays at 55-74%, the reported gain came from\n"
        "    that units mismatch and table 4.2 cannot be read as evidence for K."
    )

    out = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
                   "max_items": args.max_items, "results": results}, handle, indent=2)
    print(f"\nSaved JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
