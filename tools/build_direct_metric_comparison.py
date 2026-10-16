#!/usr/bin/env python3
"""Build a direct three-model metric-depth comparison.

This report keeps the paper's in-domain values separate from evaluations of the
released Hypersim/VKITTI checkpoints.  The latter are the only rows that can be
compared directly with the project's control and camera-aware checkpoints under
one executable protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PAPER = {
    "NYU-D": {"abs_rel": 0.056, "rmse": 0.206, "d1": 0.984},
    "KITTI": {"abs_rel": 0.045, "rmse": 1.861, "d1": 0.983},
}
EXPECTED_N = {"NYU-D": 654, "KITTI": 652}
METRICS = ("abs_rel", "rmse", "d1")


def load_checked(path: Path, dataset: str, expected_intrinsics: bool) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data["config"]
    summary = data["summary"]
    rows = data["per_image"]
    expected = EXPECTED_N[dataset]
    checks = {
        "model_type": "metric",
        "encoder": "vitl",
        "uses_camera_intrinsics": expected_intrinsics,
        "outputs_inverse_depth": False,
        "input_size": 518,
        "clip_pred": False,
    }
    for key, value in checks.items():
        if cfg.get(key) != value:
            raise ValueError(f"{path}: config[{key!r}]={cfg.get(key)!r}, expected {value!r}")
    ids = [str(row["item_id"]) for row in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique rows, got {len(ids)}")
    if summary.get("n_images") != expected or summary.get("n_skipped") != 0:
        raise ValueError(f"{path}: invalid image counts")
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite {metric}")
        if not math.isclose(float(values.mean()), float(summary[metric]), abs_tol=1e-12):
            raise ValueError(f"{path}: summary does not equal per-image mean for {metric}")
    return data


def paired_stats(proposed: dict, control: dict, seed: int = 441) -> list[dict]:
    p = {str(row["item_id"]): row for row in proposed["per_image"]}
    c = {str(row["item_id"]): row for row in control["per_image"]}
    ids = sorted(p)
    if ids != sorted(c):
        raise ValueError("control and proposed item IDs differ")
    rng = np.random.default_rng(seed)
    result = []
    for metric in METRICS:
        a = np.asarray([p[i][metric] for i in ids], dtype=np.float64)
        b = np.asarray([c[i][metric] for i in ids], dtype=np.float64)
        diff = a - b
        boots = np.asarray([
            diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10_000)
        ])
        mean = float(diff.mean())
        sd = float(diff.std(ddof=1))
        z = mean / (sd / math.sqrt(len(diff))) if sd else 0.0
        result.append({
            "metric": metric,
            "n": len(diff),
            "proposed": float(a.mean()),
            "control": float(b.mean()),
            "proposed_minus_control": mean,
            "relative_change_percent": float(100 * mean / b.mean()),
            "bootstrap_95_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "normal_approx_p": float(math.erfc(abs(z) / math.sqrt(2.0))) if sd else 1.0,
        })
    return result


def row(label: str, data: dict, status: str) -> str:
    s = data["summary"]
    return f"| {label} | {s['abs_rel']:.4f} | {s['rmse']:.4f} | {s['d1']:.4f} | {status} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/paper_metric_comparison")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "NYU-D": {
            "released": (out / "nyu_published.json", False),
            "control": (out / "nyu_control.json", False),
            "proposed": (out / "nyu_proposed.json", True),
        },
        "KITTI": {
            "released": (out / "kitti_da2_released.json", False),
            "control": (out / "kitti_control.json", False),
            "proposed": (out / "kitti_proposed_clean.json", True),
        },
    }
    data = {}
    for dataset, arms in files.items():
        data[dataset] = {name: load_checked(path, dataset, use_k) for name, (path, use_k) in arms.items()}

    report = {
        "protocol": {
            "model_type": "metric",
            "encoder": "vitl",
            "input_size": 518,
            "alignment": "none",
            "clip_pred": False,
            "NYU-D": {"n_images": 654, "crop": "eigen", "depth_field": "rawDepths"},
            "KITTI": {"n_images": 652, "crop": "none", "filelist": "datasets/raw_data/kitti/da2_val.txt"},
        },
        "paper_in_domain_reference": PAPER,
        "direct": {},
        "paired_control_vs_proposed": {},
        "limitations": [
            "The paper's in-domain metric checkpoints are not released; the paper rows are external references.",
            "Released DA2 metric rows use Hypersim (NYU-D) and Virtual KITTI 2 (KITTI) checkpoints, so they are direct protocol references but not the paper's in-domain rows.",
            "The KITTI proposed checkpoint predates cam_encoder.output_gate; evaluation used DA2_CAMERA_GATE=1.0 to restore its trained full-strength camera-token behavior.",
        ],
    }
    lines = [
        "# Direct three-model metric-depth comparison",
        "",
        "All direct rows use ViT-L, input size 518, absolute metric depth, no scale/shift alignment, and the same test protocol.",
        "",
        "## Direct released-checkpoint evaluation",
        "",
        "| Dataset / model | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Status |",
        "|---|---:|---:|---:|---|",
    ]
    csv_rows = ["dataset,model,abs_rel,rmse,d1,status"]
    for dataset in ("NYU-D", "KITTI"):
        d = data[dataset]
        report["direct"][dataset] = {k: d[k]["summary"] for k in d}
        if dataset == "KITTI":
            lines += ["## KITTI direct evaluation", "", "| Dataset / model | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Status |", "|---|---:|---:|---:|---|"]
        released_status = "released; Hypersim" if dataset == "NYU-D" else "released; Virtual KITTI 2"
        lines.append(row(f"{dataset} — DA2 released metric checkpoint", d["released"], released_status))
        csv_rows.append(f"{dataset},DA2 released metric checkpoint,{d['released']['summary']['abs_rel']:.9f},{d['released']['summary']['rmse']:.9f},{d['released']['summary']['d1']:.9f},{released_status}")
        lines.append(row(f"{dataset} — control (no K)", d["control"], "our reimplementation"))
        csv_rows.append(f"{dataset},control (no K),{d['control']['summary']['abs_rel']:.9f},{d['control']['summary']['rmse']:.9f},{d['control']['summary']['d1']:.9f},our reimplementation")
        lines.append(row(f"{dataset} — proposed (+K)", d["proposed"], "our reimplementation"))
        csv_rows.append(f"{dataset},proposed (+K),{d['proposed']['summary']['abs_rel']:.9f},{d['proposed']['summary']['rmse']:.9f},{d['proposed']['summary']['d1']:.9f},our reimplementation")
        lines.append("")
        stats = paired_stats(d["proposed"], d["control"])
        report["paired_control_vs_proposed"][dataset] = stats
        lines += [f"### {dataset}: proposed − control", "", "| Metric | Proposed | Control | Difference | Relative change | 95% bootstrap CI | p (normal approximation) |", "|---|---:|---:|---:|---:|---|---:|"]
        for s in stats:
            lines.append(f"| {s['metric']} | {s['proposed']:.6f} | {s['control']:.6f} | {s['proposed_minus_control']:+.6f} | {s['relative_change_percent']:+.2f}% | [{s['bootstrap_95_ci'][0]:+.6f}, {s['bootstrap_95_ci'][1]:+.6f}] | {s['normal_approx_p']:.3e} |")
        lines.append("")

    lines += [
        "## Published paper context (not a reproduced checkpoint)",
        "",
        "| Dataset | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Meaning |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in ("NYU-D", "KITTI"):
        p = PAPER[dataset]
        lines.append(f"| {dataset} — Depth Anything V2 paper | {p['abs_rel']:.3f} | {p['rmse']:.3f} | {p['d1']:.3f} | published in-domain fine-tuning result |")
    lines += [
        "",
        "The paper values must remain separate: the authors fine-tuned on the real target domains, while the released checkpoints evaluated above were trained on Hypersim/VKITTI and are the checkpoints we can run directly.",
        "",
    ]
    (out / "direct_three_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "direct_three_model_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "direct_three_model_comparison.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(out / "direct_three_model_comparison.md")


if __name__ == "__main__":
    main()
