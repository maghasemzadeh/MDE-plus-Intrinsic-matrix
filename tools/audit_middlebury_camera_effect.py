#!/usr/bin/env python3
"""Reproduce the Middlebury left/right-camera comparison used in the thesis.

The script runs one depth model on both rectified views of every Middlebury
scene, computes per-image AbsRel and RMSE, and uses the scene as the paired
inferential unit.  It saves the raw per-scene metrics, a machine-readable
summary, and a vector figure suitable for inclusion in the thesis.

Examples
--------
Run inference and the audit (the default checkpoint is resolved by the model
wrapper)::

    venv/bin/python tools/audit_middlebury_camera_effect.py

Recreate statistics/figures from an existing CSV without running inference::

    venv/bin/python tools/audit_middlebury_camera_effect.py --analysis-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.base import DatasetConfig
from datasets.middlebury import MiddleburyDataset
from models import create_model_wrapper
from src.metrics import compute_depth_metrics


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "middlebury_camera_audit"
DEFAULT_FIGURES = PROJECT_ROOT / "thesis" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--model-name", choices=["da2", "da2-revised"], default="da2")
    parser.add_argument("--model-type", choices=["metric", "basic"], default="metric")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vits")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--filter-regex", default=None)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_intrinsics(calib_path: Path) -> dict[str, float]:
    """Read the two 3x3 matrices without assuming fx == fy."""
    text = calib_path.read_text(encoding="utf-8")
    matrices: dict[str, list[float]] = {}
    for camera in ("cam0", "cam1"):
        match = re.search(rf"{camera}\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"{camera} missing from {calib_path}")
        values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(1))]
        if len(values) != 9:
            raise ValueError(f"Expected 9 values for {camera} in {calib_path}; got {len(values)}")
        matrices[camera] = values
    a, b = matrices["cam0"], matrices["cam1"]
    return {
        "cam0_fx": a[0], "cam0_fy": a[4], "cam0_cx": a[2], "cam0_cy": a[5],
        "cam1_fx": b[0], "cam1_fy": b[4], "cam1_cx": b[2], "cam1_cy": b[5],
    }


def run_inference(args: argparse.Namespace, csv_path: Path) -> dict[str, Any]:
    dataset = MiddleburyDataset(
        DatasetConfig(
            dataset_path=str(args.dataset_path) if args.dataset_path else None,
            regex_filter=args.filter_regex,
        )
    )
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in dataset.find_items():
        grouped[item.item_id].append(item)

    model_config = {
        "model_type": args.model_type,
        "encoder": args.encoder,
        "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
        "max_depth": args.max_depth,
        "device": args.device,
    }
    model = create_model_wrapper(args.model_name, model_config)
    checkpoint = model.get_checkpoint_path()
    rows: list[dict[str, Any]] = []

    print(f"Model checkpoint: {checkpoint}", flush=True)
    print(f"Paired scenes: {len(grouped)}", flush=True)
    for index, (scene, items) in enumerate(sorted(grouped.items()), start=1):
        by_camera = {item.camera_id: item for item in items}
        if set(by_camera) != {"0", "1"}:
            print(f"[{index}/{len(grouped)}] skip {scene}: incomplete stereo pair", flush=True)
            continue
        row: dict[str, Any] = {"scene": scene}
        row.update(_read_intrinsics(Path(items[0].metadata["scene_path"]) / "calib.txt"))

        failed = False
        for camera in ("0", "1"):
            item = by_camera[camera]
            image = cv2.imread(item.image_path)
            if image is None:
                print(f"[{index}/{len(grouped)}] skip {scene}: unreadable camera {camera}", flush=True)
                failed = True
                break
            gt = dataset.load_gt_depth(item.gt_path, item)
            pred = np.asarray(model.infer_image(image, input_size=args.input_size), dtype=np.float32).squeeze()
            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch for {scene}/cam{camera}: pred={pred.shape}, gt={gt.shape}")
            metrics = compute_depth_metrics(
                pred,
                gt,
                is_metric_model=model.is_metric(),
                outputs_inverse_depth=model.outputs_inverse_depth(),
            )
            row[f"cam{camera}_abs_rel"] = metrics["abs_rel"]
            row[f"cam{camera}_rmse"] = metrics["rmse"]
            row[f"cam{camera}_n_valid"] = metrics["n_valid"]
        if not failed:
            rows.append(row)
            print(
                f"[{index}/{len(grouped)}] {scene}: "
                f"AbsRel {row['cam0_abs_rel']:.4f}/{row['cam1_abs_rel']:.4f}; "
                f"RMSE {row['cam0_rmse']:.4f}/{row['cam1_rmse']:.4f} m",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No complete Middlebury pairs were evaluated")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "encoder": model.encoder,
        "checkpoint": str(checkpoint),
        "max_depth": model.max_depth,
        "input_size": args.input_size,
        "device": args.device,
    }


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [key for key in rows[0] if key != "scene"]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def bootstrap_mean_ci(diff: np.ndarray, seed: int = 20260805) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(20_000, len(diff)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def metric_statistics(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    cam0 = np.asarray([row[f"cam0_{metric}"] for row in rows], dtype=float)
    cam1 = np.asarray([row[f"cam1_{metric}"] for row in rows], dtype=float)
    diff = cam0 - cam1
    t_stat, t_p = ttest_rel(cam0, cam1)
    try:
        w_stat, w_p = wilcoxon(diff, alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    ci_low, ci_high = bootstrap_mean_ci(diff)
    return {
        "n_pairs": len(diff),
        "cam0_mean": float(cam0.mean()),
        "cam0_sd": float(cam0.std(ddof=1)),
        "cam1_mean": float(cam1.mean()),
        "cam1_sd": float(cam1.std(ddof=1)),
        "mean_difference_cam0_minus_cam1": float(diff.mean()),
        "bootstrap_95_ci": [ci_low, ci_high],
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
    }


def calibration_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = {}
    for name in ("fx", "fy", "cx", "cy"):
        values = np.asarray([row[f"cam1_{name}"] - row[f"cam0_{name}"] for row in rows])
        deltas[name] = {
            "all_equal": bool(np.allclose(values, 0.0)),
            "mean_cam1_minus_cam0": float(values.mean()),
            "range_cam1_minus_cam0": [float(values.min()), float(values.max())],
        }
    return deltas


def average_rectifications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average perfect/imperfect rectifications of the same scene content."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base_scene = re.sub(r"-(?:imperfect|perfect)$", "", str(row["scene"]))
        grouped[base_scene].append(row)
    averaged = []
    for scene, variants in sorted(grouped.items()):
        entry: dict[str, Any] = {"scene": scene, "n_rectifications": len(variants)}
        for key in variants[0]:
            if key != "scene":
                entry[key] = float(np.mean([float(row[key]) for row in variants]))
        averaged.append(entry)
    return averaged


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, Any]], output_pdf: Path, output_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    colors = {"abs_rel": "#2B6CB0", "rmse": "#C05621"}
    labels = {"abs_rel": "AbsRel", "rmse": "RMSE (m)"}
    for col, metric in enumerate(("abs_rel", "rmse")):
        cam0 = np.asarray([row[f"cam0_{metric}"] for row in rows])
        cam1 = np.asarray([row[f"cam1_{metric}"] for row in rows])
        diff = cam0 - cam1

        ax = axes[0, col]
        for left, right in zip(cam0, cam1):
            ax.plot([0, 1], [left, right], color="#A0AEC0", alpha=0.42, linewidth=0.8)
        ax.scatter(np.zeros_like(cam0), cam0, s=17, color=colors[metric], alpha=0.72, zorder=3)
        ax.scatter(np.ones_like(cam1), cam1, s=17, color=colors[metric], alpha=0.72, zorder=3)
        ax.plot([0, 1], [cam0.mean(), cam1.mean()], color="#1A202C", marker="D", linewidth=2.2, zorder=4)
        ax.set_xticks([0, 1], ["Camera 0", "Camera 1"])
        ax.set_ylabel(labels[metric])
        ax.set_title(f"Paired scene values: {labels[metric]}")
        ax.grid(axis="y", alpha=0.22)

        order = np.argsort(diff)
        ax = axes[1, col]
        y = np.arange(len(diff))
        ax.barh(y, diff[order], color=np.where(diff[order] >= 0, colors[metric], "#718096"), alpha=0.82)
        ax.axvline(0, color="#1A202C", linewidth=1)
        ax.axvline(diff.mean(), color="#C53030", linewidth=2, linestyle="--", label="Mean difference")
        ax.set_yticks([])
        ax.set_xlabel(f"Camera 0 - Camera 1 ({labels[metric]})")
        ax.set_title("Ordered paired differences")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="x", alpha=0.22)

    fig.suptitle("Middlebury left/right-camera audit (23 independent scene contents)", fontsize=14)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    csv_path = results_dir / "per_scene_metrics.csv"
    metadata_path = results_dir / "run_metadata.json"

    if args.analysis_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"Analysis CSV does not exist: {csv_path}")
    elif csv_path.exists() and not args.force:
        print(f"Using existing inference results: {csv_path}", flush=True)
    else:
        metadata = run_inference(args, csv_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = load_rows(csv_path)
    base_rows = average_rectifications(rows)
    write_rows(base_rows, results_dir / "per_base_scene_metrics.csv")
    summary = {
        "protocol": (
            "two-sided paired tests; perfect/imperfect rectifications are first averaged "
            "so the 23 independent scene contents are the primary inferential units"
        ),
        "calibration": calibration_statistics(rows),
        "primary_base_scene_analysis": {
            "abs_rel": metric_statistics(base_rows, "abs_rel"),
            "rmse": metric_statistics(base_rows, "rmse"),
        },
        "sensitivity_analysis_all_46_rectifications": {
            "abs_rel": metric_statistics(rows, "abs_rel"),
            "rmse": metric_statistics(rows, "rmse"),
        },
        "limitations": [
            "The paired images are different viewpoints, so a camera-side effect is not a causal intrinsic-parameter effect.",
            "Within each stereo pair, fx and fy are identical; the systematic matrix difference is cx.",
            "Perfect and imperfect rectifications of the same base scene are related observations.",
        ],
    }
    summary_path = results_dir / "statistical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(
        base_rows,
        figure_dir / "middlebury_camera_audit.pdf",
        results_dir / "middlebury_camera_audit.png",
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Raw metrics: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Thesis figure: {figure_dir / 'middlebury_camera_audit.pdf'}")


if __name__ == "__main__":
    main()
