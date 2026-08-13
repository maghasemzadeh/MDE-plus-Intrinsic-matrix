"""Regenerate and independently validate thesis section 4.5.7.

Section 4.5.7 reports the third controlled-training round: metric ViT-L
models trained with focal jitter (m=1.6), where the proposed arm receives
camera intrinsics and the matched control does not.  The primary evaluation
is absolute depth on the 654-image NYUv2 Eigen test split, with no scale or
shift alignment.

Run from the repository root on the training/evaluation machine::

    venv/bin/python tools/validate_section_4_5_7.py --regenerate

To regenerate the K-sweep and field-of-view diagnostics as well::

    venv/bin/python tools/validate_section_4_5_7.py \
        --regenerate --regenerate-diagnostics

Without regeneration, the script audits JSON files already present in the
input directory.  It recomputes means from per-image rows, paired bootstrap
confidence intervals, Wilcoxon tests, Holm-adjusted p-values, and the
diagnostic summaries.  It writes machine-readable JSON and a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


EXPECTED_N = 654
PRIMARY_METRICS = ("abs_rel", "rmse", "d1", "silog", "log10")
DISPLAY = {
    "abs_rel": "AbsRel",
    "rmse": "RMSE",
    "d1": "delta_1",
    "silog": "SILog",
    "log10": "log10",
}
HIGHER_IS_BETTER = {"d1"}


@dataclass(frozen=True)
class Arm:
    key: str
    filename: str
    checkpoint: str
    uses_intrinsics: bool
    thesis_means: dict[str, float]


ARMS = (
    Arm(
        "proposed",
        "nyu_METRIC_REVISED_vitl_focal_fresh.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_focal/"
        "revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260709_214151/best.pth",
        True,
        {
            "abs_rel": 0.0849,
            "rmse": 0.3071,
            "d1": 0.9520,
            "silog": 0.0942,
            "log10": 0.0363,
        },
    ),
    Arm(
        "control",
        "nyu_METRIC_BASEFT_vitl_focal_fresh.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_focal/"
        "base_metric_vitl_nyu_lr5e-6_bs2_20260709_233640/best.pth",
        False,
        {
            "abs_rel": 0.1030,
            "rmse": 0.3431,
            "d1": 0.9226,
            "silog": 0.1045,
            "log10": 0.0427,
        },
    ),
)

THESIS_DIFFS = {
    "abs_rel": (-0.0182, -0.0210, -0.0153, 4.0e-31),
    "rmse": (-0.0361, -0.0432, -0.0289, 1.0e-24),
    "d1": (+0.0294, +0.0224, +0.0367, 2.5e-24),
    "silog": (-0.0102, -0.0118, -0.0087, 1.5e-34),
    "log10": (-0.0065, -0.0076, -0.0053, 2.8e-27),
}

K_SWEEP_FILENAME = "k_sweep_section_4_5_7_fresh.json"
FOV_FILENAME = "fov_varied_section_4_5_7_fresh.json"
EXPECTED_FACTORS = [0.6, 0.8, 1.0, 1.25, 1.6]
EXPECTED_ZOOMS = [1.0, 1.4, 1.8]
THESIS_FOV_ABSREL = {
    "proposed": {"1": 0.0924, "1.4": 0.0974, "1.8": 0.1065},
    "control": {"1": 0.1102, "1.4": 0.1100, "1.8": 0.1148},
}


def regenerate(root: Path, output_dir: Path, diagnostics: bool, force: bool = False) -> None:
    """Regenerate the two primary evaluations and optional diagnostics."""
    for arm in ARMS:
        output_path = output_dir / arm.filename
        if output_path.is_file() and not force:
            print(f"\nReusing completed {arm.key} evaluation: {output_path}", flush=True)
            continue
        checkpoint = root / arm.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        command = [
            sys.executable,
            str(root / "evaluate_paper.py"),
            "--dataset", "nyu",
            "--model", "da2-revised",
            "--model-type", "metric",
            "--encoder", "vitl",
            "--checkpoint", str(checkpoint),
            "--crop", "eigen",
            "--clip-pred", "false",
            "--output-json", str(output_path),
        ]
        print(f"\nRegenerating {arm.key} evaluation", flush=True)
        subprocess.run(command, cwd=root, check=True)

    if not diagnostics:
        return

    checkpoint_args = []
    for arm in ARMS:
        checkpoint_args.extend(
            ["--checkpoint", f"{arm.key}={root / arm.checkpoint}"]
        )
    sweep_path = output_dir / K_SWEEP_FILENAME
    sweep_command = [
        sys.executable,
        str(root / "tools" / "k_sweep.py"),
        *checkpoint_args,
        "--encoder", "vitl",
        "--model-type", "metric",
        "--num-images", "20",
        "--factors", *[str(x) for x in EXPECTED_FACTORS],
        "--output-dir", str(output_dir),
        "--tag", "section_4_5_7_fresh",
    ]
    if sweep_path.is_file() and not force:
        print(f"\nReusing completed K-sweep diagnostic: {sweep_path}", flush=True)
    else:
        print("\nRegenerating K-sweep diagnostic", flush=True)
        subprocess.run(sweep_command, cwd=root, check=True)

    fov_command = [
        sys.executable,
        str(root / "tools" / "eval_fov_varied.py"),
        *checkpoint_args,
        "--encoder", "vitl",
        "--model-type", "metric",
        "--zooms", *[str(x) for x in EXPECTED_ZOOMS],
        "--output-dir", str(output_dir),
        "--tag", "section_4_5_7_fresh",
    ]
    fov_path = output_dir / FOV_FILENAME
    if fov_path.is_file() and not force:
        print(f"\nReusing completed field-of-view diagnostic: {fov_path}", flush=True)
    else:
        print("\nRegenerating field-of-view diagnostic", flush=True)
        subprocess.run(fov_command, cwd=root, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_primary(path: Path, arm: Arm) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = load_json(path)
    for field in ("summary", "config", "per_image"):
        if field not in data:
            raise ValueError(f"{path}: missing top-level field {field!r}")

    expected_config = {
        "dataset": "nyu",
        "nyu_depth_field": "rawDepths",
        "model_type": "metric",
        "encoder": "vitl",
        "uses_camera_intrinsics": arm.uses_intrinsics,
        "input_size": 518,
        "min_depth_eval": 0.001,
        "max_depth_eval": 10.0,
        "crop": "eigen",
        "clip_pred": False,
        "outputs_inverse_depth": False,
    }
    for key, wanted in expected_config.items():
        actual = data["config"].get(key)
        if actual != wanted:
            raise ValueError(f"{path}: config[{key!r}]={actual!r}, expected {wanted!r}")

    rows = data["per_image"]
    indexed = {str(row["item_id"]): row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"{path}: duplicate item_id values")
    if len(indexed) != EXPECTED_N:
        raise ValueError(f"{path}: expected {EXPECTED_N} images, found {len(indexed)}")
    if int(data["summary"].get("n_images", -1)) != EXPECTED_N:
        raise ValueError(f"{path}: summary n_images is not {EXPECTED_N}")
    if int(data["summary"].get("n_skipped", -1)) != 0:
        raise ValueError(f"{path}: expected zero skipped images")

    for metric in PRIMARY_METRICS:
        values = np.asarray([row.get(metric) for row in indexed.values()], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite per-image {metric}")
        recomputed = float(values.mean())
        stored = float(data["summary"][metric])
        if not np.isclose(recomputed, stored, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: stored {metric}={stored} != recomputed {recomputed}")
        if round(recomputed, 4) != arm.thesis_means[metric]:
            raise AssertionError(
                f"{path}: {metric} rounds to {round(recomputed, 4)}, "
                f"but thesis prints {arm.thesis_means[metric]}"
            )
    return data, indexed


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def displayed_p_agrees(actual: float, printed: float) -> bool:
    return bool(np.isclose(actual, printed, rtol=0.08, atol=0.0))


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    total = len(values)
    for rank, original in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[original]))
        adjusted[original] = running
    return adjusted.tolist()


def validate_k_sweep(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("config", {}).get("factors") != EXPECTED_FACTORS:
        raise ValueError(f"{path}: unexpected focal factors")
    if int(data["config"].get("num_images", -1)) != 20:
        raise ValueError(f"{path}: K-sweep must use 20 images")
    output: dict[str, Any] = {}
    for arm in ARMS:
        result = data["results"][arm.key]
        if bool(result["uses_camera_intrinsics"]) != arm.uses_intrinsics:
            raise ValueError(f"{path}: uses_camera_intrinsics mismatch for {arm.key}")
        rows = result["per_image"]
        if len(rows) != 20:
            raise ValueError(f"{path}: expected 20 K-sweep rows for {arm.key}")
        slopes = np.asarray([row["slope_median"] for row in rows], dtype=np.float64)
        mean, std = float(slopes.mean()), float(slopes.std())
        if not np.isclose(mean, result["slope_median_mean"], atol=1e-12):
            raise ValueError(f"{path}: stored K-sweep mean mismatch for {arm.key}")
        if not np.isclose(std, result["slope_median_std"], atol=1e-12):
            raise ValueError(f"{path}: stored K-sweep std mismatch for {arm.key}")
        if arm.key == "proposed" and not (0.85 <= mean <= 1.05):
            raise AssertionError(f"{path}: proposed K-sweep slope {mean} is not near 1")
        if arm.key == "control" and abs(mean) > 0.05:
            raise AssertionError(f"{path}: control K-sweep slope {mean} is not near 0")
        output[arm.key] = {"slope_mean": mean, "slope_std": std, "n_images": len(rows)}
    return output


def validate_fov(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if [float(x) for x in data.get("config", {}).get("zooms", [])] != EXPECTED_ZOOMS:
        raise ValueError(f"{path}: unexpected zoom factors")
    output: dict[str, Any] = {}
    for arm in ARMS:
        result = data["results"][arm.key]
        if bool(result["uses_camera_intrinsics"]) != arm.uses_intrinsics:
            raise ValueError(f"{path}: uses_camera_intrinsics mismatch for {arm.key}")
        output[arm.key] = {}
        for zoom in ("1", "1.4", "1.8"):
            rows = result["per_zoom"][zoom]
            if len(rows) != EXPECTED_N:
                raise ValueError(f"{path}: {arm.key}/{zoom} has {len(rows)} images")
            recomputed = float(np.mean([row["abs_rel"] for row in rows]))
            stored = float(result["summary"][zoom]["abs_rel"])
            if not np.isclose(recomputed, stored, atol=1e-12):
                raise ValueError(f"{path}: stored FoV mean mismatch for {arm.key}/{zoom}")
            if round(recomputed, 4) != THESIS_FOV_ABSREL[arm.key][zoom]:
                raise AssertionError(
                    f"{path}: {arm.key}/{zoom} AbsRel rounds to {round(recomputed, 4)}, "
                    f"thesis reports {THESIS_FOV_ABSREL[arm.key][zoom]}"
                )
            output[arm.key][zoom] = recomputed
    for zoom in ("1", "1.4", "1.8"):
        if output["proposed"][zoom] >= output["control"][zoom]:
            raise AssertionError(f"{path}: proposed model does not win at zoom {zoom}")
    return output


def validate(input_dir: Path) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    common_ids: set[str] | None = None
    aggregates = []
    for arm in ARMS:
        data, indexed = load_primary(input_dir / arm.filename, arm)
        ids = set(indexed)
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError(f"{arm.filename}: item_id set does not match the paired arm")
        loaded[arm.key] = (data, indexed)
        aggregates.append(
            {
                "arm": arm.key,
                **{metric: float(data["summary"][metric]) for metric in PRIMARY_METRICS},
                "n_images": len(indexed),
            }
        )

    item_ids = sorted(common_ids or [])
    paired_rows = []
    raw_p = []
    for metric in PRIMARY_METRICS:
        proposed = np.asarray(
            [loaded["proposed"][1][item][metric] for item in item_ids], dtype=np.float64
        )
        control = np.asarray(
            [loaded["control"][1][item][metric] for item in item_ids], dtype=np.float64
        )
        diff = proposed - control
        lo, hi = bootstrap_ci(diff)
        p_value = float(wilcoxon(proposed, control).pvalue)
        printed_diff, printed_lo, printed_hi, printed_p = THESIS_DIFFS[metric]
        if round(float(diff.mean()), 4) != printed_diff:
            raise AssertionError(f"{metric}: mean difference does not reproduce thesis")
        if round(lo, 4) != printed_lo or round(hi, 4) != printed_hi:
            raise AssertionError(f"{metric}: bootstrap CI does not reproduce thesis")
        if not displayed_p_agrees(p_value, printed_p):
            raise AssertionError(f"{metric}: Wilcoxon p={p_value} != printed {printed_p}")
        benefit = float(diff.mean()) if metric in HIGHER_IS_BETTER else float(-diff.mean())
        relative_benefit = (
            benefit / float(control.mean()) * 100.0
            if metric not in HIGHER_IS_BETTER
            else benefit / float(control.mean()) * 100.0
        )
        paired_rows.append(
            {
                "metric": DISPLAY[metric],
                "metric_key": metric,
                "proposed_mean": float(proposed.mean()),
                "control_mean": float(control.mean()),
                "difference_proposed_minus_control": float(diff.mean()),
                "ci95": [lo, hi],
                "wilcoxon_p": p_value,
                "wilcoxon_holm_p": None,
                "benefit_percent_of_control": relative_benefit,
            }
        )
        raw_p.append(p_value)

    for row, adjusted in zip(paired_rows, holm_adjust(raw_p)):
        row["wilcoxon_holm_p"] = adjusted
        if adjusted >= 0.05:
            raise AssertionError(f"{row['metric']}: not significant after Holm correction")

    diagnostics: dict[str, Any] = {}
    sweep_path = input_dir / K_SWEEP_FILENAME
    fov_path = input_dir / FOV_FILENAME
    if sweep_path.is_file():
        diagnostics["k_sweep"] = validate_k_sweep(sweep_path)
    if fov_path.is_file():
        diagnostics["fov_varied"] = validate_fov(fov_path)

    return {
        "status": "PASS",
        "protocol": (
            "NYUv2 Eigen-654 rawDepths; metric ViT-L; eigen crop; [0.001, 10] m; "
            "no per-image scale/shift alignment; paired by item_id"
        ),
        "checks": {
            "primary_files": len(ARMS),
            "paired_images": len(item_ids),
            "aggregate_cells": len(ARMS) * len(PRIMARY_METRICS),
            "paired_metrics": len(PRIMARY_METRICS),
            "diagnostics_present": sorted(diagnostics),
        },
        "aggregates": aggregates,
        "paired_tests": paired_rows,
        "diagnostics": diagnostics,
        "validity_caveat": (
            "Per-image pairing establishes a precise test-set difference for these checkpoints, "
            "but it does not estimate variation across independent training seeds."
        ),
    }


def fmt_p(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-3 else f"{value:.3e}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Section 4.5.7 focal-jitter validation",
        "",
        f"Overall status: **{report['status']}**",
        "",
        f"Protocol: {report['protocol']}",
        "",
        "## Recomputed primary results",
        "",
        "| Metric | Proposed (+K) | Control | Proposed - control | Bootstrap 95% CI | Wilcoxon p | Holm p | Benefit vs control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["paired_tests"]:
        lines.append(
            f"| {row['metric']} | {row['proposed_mean']:.9f} | {row['control_mean']:.9f} | "
            f"{row['difference_proposed_minus_control']:+.9f} | "
            f"[{row['ci95'][0]:+.9f}, {row['ci95'][1]:+.9f}] | "
            f"{fmt_p(row['wilcoxon_p'])} | {fmt_p(row['wilcoxon_holm_p'])} | "
            f"{row['benefit_percent_of_control']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All five confidence intervals exclude zero and all five results remain significant "
            "after Holm family-wise-error correction.",
        ]
    )

    diagnostics = report["diagnostics"]
    if "k_sweep" in diagnostics:
        lines.extend(["", "## K-sweep diagnostic", "", "| Arm | Slope mean | Slope SD | Images |", "|---|---:|---:|---:|"])
        for arm in ("proposed", "control"):
            row = diagnostics["k_sweep"][arm]
            lines.append(
                f"| {arm} | {row['slope_mean']:.6f} | {row['slope_std']:.6f} | {row['n_images']} |"
            )
    if "fov_varied" in diagnostics:
        lines.extend(["", "## Field-of-view diagnostic (AbsRel)", "", "| Arm | zoom 1.0 | zoom 1.4 | zoom 1.8 |", "|---|---:|---:|---:|"])
        for arm in ("proposed", "control"):
            row = diagnostics["fov_varied"][arm]
            lines.append(f"| {arm} | {row['1']:.9f} | {row['1.4']:.9f} | {row['1.8']:.9f} |")

    lines.extend(
        [
            "",
            "## Validity conclusion",
            "",
            "- The stored summaries exactly equal means recomputed from 654 finite, uniquely paired per-image rows.",
            "- Every thesis aggregate, paired difference, bootstrap interval, and Wilcoxon p-value reproduces at its displayed precision.",
            "- The evaluation metadata confirms metric depth with no alignment, the Eigen crop, raw NYU depth, and matched model capacity.",
            f"- Caveat: {report['validity_caveat']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/section_4_5_7_validation"),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rerun evaluate_paper.py for both focal-jitter checkpoints",
    )
    parser.add_argument(
        "--regenerate-diagnostics",
        action="store_true",
        help="also rerun the 20-image K-sweep and full 654-image FoV evaluation",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="overwrite already completed fresh outputs instead of resuming",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    if args.regenerate_diagnostics and not args.regenerate:
        parser.error("--regenerate-diagnostics requires --regenerate")

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    if args.regenerate:
        regenerate(
            root,
            input_dir,
            diagnostics=args.regenerate_diagnostics,
            force=args.force_regenerate,
        )

    report = validate(input_dir)
    markdown = build_markdown(report)
    json_out = args.json_out or input_dir / "validation_report.json"
    markdown_out = args.markdown_out or input_dir / "validation_report.md"
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
