#!/usr/bin/env python3
"""Report completed GPU pilot summaries without inventing per-image evidence."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/paper_metric_comparison"
OUT = BASE / "kitti_pilot_20261002"
METRICS = ("abs_rel", "rmse", "d1")


def validate_summary(summary):
    assert summary["n_images"] == 652 and summary["n_skipped"] == 0
    assert all(math.isfinite(summary[k]) for k in METRICS)
    assert summary["abs_rel"] >= 0 and summary["rmse"] >= 0
    assert 0 <= summary["d1"] <= 1


def load_reference(name):
    data = json.loads((BASE / name).read_text())
    validate_summary(data["summary"])
    cfg = data["config"]
    for key, value in {"dataset": "kitti", "model_type": "metric", "encoder": "vitl",
                       "input_size": 518, "crop": "none", "clip_pred": False,
                       "outputs_inverse_depth": False,
                       "min_depth_eval": 0.001, "max_depth_eval": 80.0}.items():
        assert cfg[key] == value, (name, key, cfg[key])
    rows = data["per_image"]
    assert len(rows) == len({r["item_id"] for r in rows}) == 652
    for key in METRICS:
        assert math.isclose(sum(r[key] for r in rows) / 652, data["summary"][key], abs_tol=1e-12)
    return data["summary"]


def main():
    capture = json.loads((OUT / "pilot_summary_capture.json").read_text())
    rows = {
        "Released Depth Anything V2 metric (VKITTI)": load_reference("kitti_da2_released.json"),
        "Previous control": load_reference("kitti_control.json"),
        "Previous proposed (+K; legacy gate=1)": load_reference("kitti_proposed_clean.json"),
        "New control pilot (no K)": capture["control"]["summary"],
        "New proposed pilot (+K)": capture["proposed"]["summary"],
    }
    for summary in rows.values():
        validate_summary(summary)
    p = capture["proposed"]["summary"]
    changes = {}
    for label, ref in rows.items():
        if label.startswith("New proposed"):
            continue
        changes[label] = {
            "abs_rel_change_percent": 100 * (p["abs_rel"] / ref["abs_rel"] - 1),
            "rmse_change_percent": 100 * (p["rmse"] / ref["rmse"] - 1),
            "d1_change_percentage_points": 100 * (p["d1"] - ref["d1"]),
        }
    report = {"status": "completed exploratory pilot; corrected-head follow-up not run",
              "provenance": capture["provenance"], "protocol": capture["protocol"],
              "training": capture["training"], "results": rows,
              "new_proposed_changes": changes, "limitations": capture["limitations"]}
    (OUT / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Completed KITTI metric-depth pilot",
        "",
        "The new camera-aware pilot improved on its matched control and the previous proposed checkpoint. It did **not** outperform the released Depth Anything V2 metric model.",
        "",
        "All rows were evaluated on the same 652-image KITTI file list, with zero skipped images: ViT-L, input size 518, 0.001–80 m ground-truth range, no crop, no scale/shift alignment, and no prediction clipping. These are our measured evaluations, not copied paper values.",
        "",
        "| Model | AbsRel ↓ | RMSE (m) ↓ | δ1 ↑ |",
        "|---|---:|---:|---:|",
    ]
    for label, summary in rows.items():
        lines.append(f"| {label} | {summary['abs_rel']:.4f} | {summary['rmse']:.4f} | {summary['d1']:.4f} |")
    lines += ["", "## Improvement of the new proposed pilot", "",
              "| Reference | AbsRel change | RMSE change | δ1 change (percentage points) |",
              "|---|---:|---:|---:|"]
    for label, values in changes.items():
        lines.append(f"| {label} | {values['abs_rel_change_percent']:+.2f}% | {values['rmse_change_percent']:+.2f}% | {values['d1_change_percentage_points']:+.2f} |")
    lines += [
        "", "Negative error changes and positive δ1 changes are improvements.",
        "", "## What was trained", "",
        "Both new arms ran one full epoch (1,250 minibatches, 625 optimizer steps) over the same 5,000 VKITTI2 samples. Scene18 supplied 1,000 fine-tuning-validation samples; training used Scene01, Scene02, Scene06 and Scene20. No real KITTI evaluation images were used for this training.",
        "",
        "Shared settings: seed 42, batch size 4, gradient accumulation 2, AdamW, LR 5e-6, head multiplier 1, gradient clipping 1, SiLog lambda 0.5, frozen DINOv2 and one warm-up epoch. This pilot therefore used a warm-up-only schedule, not a completed long-run decay schedule. The proposed arm additionally enabled K and injected its camera token at layer 0. Final-epoch checkpoints were evaluated.",
        "",
        "## Important qualification", "",
        "The runtime audit showed train.py imported the ReLU output implementation even with --model-type metric. Both pilots learned direct depth in metres and were evaluated without alignment, so the reported errors are metric-depth measurements. However, neither preserves the official checkpoint's sigmoid-times-80 output at initialization. This is an exploratory comparison of our two implementations, not a controlled ablation of the unchanged released metric architecture.",
        "",
    ]
    lines += [f"- {note}" for note in capture["limitations"]]
    lines += ["", "## Artifact provenance and preservation", "",
              "The two new summary rows are exact captures of completed remote JSON summaries printed by tools in this conversation. Interrupted copy requests did not bring the original per-image JSONs or new logs into this local directory. No paired statistics are claimed. Older reference JSONs were checked locally for counts, unique IDs, protocol settings and per-image means.",
              "",
              "Remote root: /home/ubuntu/projects/MDE-plus-Intrinsic-matrix. The new checkpoints remain only on that server:", ""]
    for arm in ("control", "proposed"):
        lines += [f"- {arm}: `{capture[arm]['checkpoint']}`",
                  f"- {arm} raw evaluation: `{capture[arm]['evaluation_json']}`"]
    lines += ["", "No thesis, slide or prior checkpoint was edited for this experiment. New local artifacts are confined to experiment/report tooling and result files. The native sigmoid-head follow-up and a publication-grade calibrated-K ablation remain unfinished.", ""]
    (OUT / "comparison_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"Saved report: {OUT / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
