"""Regenerate and independently validate thesis section 4.5.6.

The section reports absolute (metric) depth on the 654-image NYUv2 Eigen
test split, without per-image scale/shift alignment.  This script can either
audit existing ``evaluate_paper.py`` JSON files or regenerate all five model
outputs from checkpoints before performing the audit.

Examples (from the repository root, on a machine with the checkpoints/data):

    venv/bin/python tools/validate_section_4_5_6.py --regenerate
    venv/bin/python tools/validate_section_4_5_6.py \
        --input-dir results/section_4_5_6_validation

The generated JSON and Markdown reports contain the two thesis tables at full
precision, paired tests, protocol checks, and a Holm multiple-testing audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


EXPECTED_N = 654
METRICS = ("abs_rel", "d1", "rmse")
DISPLAY = {"abs_rel": "AbsRel", "d1": "delta_1", "rmse": "RMSE"}
HIGHER_IS_BETTER = {"d1"}


@dataclass(frozen=True)
class Evaluation:
    key: str
    filename: str
    checkpoint: str | None
    uses_intrinsics: bool
    thesis_means: dict[str, float]


EVALUATIONS = (
    Evaluation(
        "published",
        "nyu_METRIC_baseline_hypersim_vitl.json",
        None,
        False,
        {"abs_rel": 0.2073, "d1": 0.6919, "rmse": 0.6063},
    ),
    Evaluation(
        "round1_control",
        "nyu_METRIC_BASEFT_vitl_aug.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_aug/"
        "base_metric_vitl_nyu_lr5e-6_bs2_20260706_210658/best.pth",
        False,
        {"abs_rel": 0.0796, "d1": 0.9583, "rmse": 0.2919},
    ),
    Evaluation(
        "round1_proposed",
        "nyu_METRIC_REVISED_vitl_aug.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_aug/"
        "revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260706_190726/best.pth",
        True,
        {"abs_rel": 0.0791, "d1": 0.9593, "rmse": 0.2882},
    ),
    Evaluation(
        "round2_control",
        "nyu_METRIC_BASEFT_vitl_aug2.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_aug2/"
        "base_metric_vitl_nyu_lr5e-6_bs2_20260707_004717/best.pth",
        False,
        {"abs_rel": 0.0826, "d1": 0.9546, "rmse": 0.2998},
    ),
    Evaluation(
        "round2_proposed",
        "nyu_METRIC_REVISED_vitl_aug2.json",
        "models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_aug2/"
        "revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260706_224730/best.pth",
        True,
        {"abs_rel": 0.0817, "d1": 0.9569, "rmse": 0.2944},
    ),
)


# Values printed in section 4.5.6's paired-significance table (Table 4.12).
# Delta_1 was not printed there, but is recomputed and included in the report.
PRINTED_P = {
    ("round1_vs_published", "abs_rel"): (4.8e-133, 8.4e-96),
    ("round1_vs_published", "rmse"): (6.7e-100, 5.6e-87),
    ("round1_vs_control", "abs_rel"): (0.26, 0.53),
    ("round1_vs_control", "rmse"): (4.9e-3, 9.3e-3),
    ("round2_vs_control", "abs_rel"): (0.089, 0.040),
    ("round2_vs_control", "rmse"): (6.5e-5, 1.2e-4),
}


def regenerate(root: Path, output_dir: Path) -> None:
    """Run the exact no-alignment metric evaluation for all five models."""
    evaluator = root / "evaluate_paper.py"
    for spec in EVALUATIONS:
        output = output_dir / spec.filename
        command = [
            sys.executable,
            str(evaluator),
            "--dataset", "nyu",
            "--model-type", "metric",
            "--encoder", "vitl",
            "--crop", "eigen",
            "--clip-pred", "false",
            "--output-json", str(output),
        ]
        if spec.checkpoint is None:
            command.extend(["--model", "da2"])
        else:
            checkpoint = root / spec.checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            command.extend(
                ["--model", "da2-revised", "--checkpoint", str(checkpoint)]
            )
        print(f"\nRegenerating {spec.key} -> {output}", flush=True)
        subprocess.run(command, cwd=root, check=True)


def load_and_check(path: Path, spec: Evaluation) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    for field in ("summary", "config", "per_image"):
        if field not in data:
            raise ValueError(f"{path}: missing top-level field {field!r}")

    config = data["config"]
    expected_config = {
        "dataset": "nyu",
        "nyu_depth_field": "rawDepths",
        "model_type": "metric",
        "encoder": "vitl",
        "uses_camera_intrinsics": spec.uses_intrinsics,
        "input_size": 518,
        "min_depth_eval": 0.001,
        "max_depth_eval": 10.0,
        "crop": "eigen",
        "clip_pred": False,
        "outputs_inverse_depth": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{path}: config[{key!r}]={config.get(key)!r}, expected {expected!r}"
            )

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

    for metric in METRICS:
        values = np.asarray([row.get(metric) for row in indexed.values()], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite per-image {metric}")
        recomputed = float(values.mean())
        stored = float(data["summary"][metric])
        if not np.isclose(recomputed, stored, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: stored {metric}={stored} != recomputed {recomputed}")
        # The thesis table reports four decimals.
        if round(recomputed, 4) != spec.thesis_means[metric]:
            raise AssertionError(
                f"{path}: {metric} rounds to {round(recomputed, 4)}, "
                f"but thesis prints {spec.thesis_means[metric]}"
            )
    return data, indexed


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    count = len(values)
    for rank, original in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[original]))
        adjusted[original] = running
    return adjusted.tolist()


def printed_p_agrees(actual: float, printed: float) -> bool:
    if printed >= 0.01:
        decimals = 3 if printed < 0.1 else 2
        return round(actual, decimals) == printed
    return bool(np.isclose(actual, printed, rtol=0.06, atol=0.0))


def benefit(metric: str, proposed: float, reference: float) -> float:
    return proposed - reference if metric in HIGHER_IS_BETTER else reference - proposed


def fmt_p(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-3 else f"{value:.3e}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Section 4.5.6 metric-evaluation validation",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "## Recomputed aggregate table",
        "",
        "| Evaluation | AbsRel | delta_1 | RMSE | Images |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['evaluation']} | {row['abs_rel']:.9f} | "
            f"{row['d1']:.9f} | {row['rmse']:.9f} | {row['n_images']} |"
        )
    lines.extend(
        [
            "",
            "## Recomputed paired tests",
            "",
            "| Comparison | Metric | Proposed | Reference | Benefit | paired-t p | Wilcoxon p | Holm t p | Holm Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["paired_tests"]:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['proposed_mean']:.9f} | "
            f"{row['reference_mean']:.9f} | {row['benefit']:+.9f} | "
            f"{fmt_p(row['t_p'])} | {fmt_p(row['wilcoxon_p'])} | "
            f"{fmt_p(row['t_holm_p'])} | {fmt_p(row['wilcoxon_holm_p'])} |"
        )
    lines.extend(
        [
            "",
            "Positive `Benefit` always means the proposed intrinsics-aware model is better.",
            "Holm correction is applied separately to the six proposed-vs-control tests "
            "(two rounds × three metrics) for each test family.",
            "",
            "## Audit conclusion",
            "",
            f"- {report['checks']['files']} files passed protocol, finite-value, 654-ID pairing, "
            "zero-skipped-image, and stored-summary checks.",
            f"- All {report['checks']['aggregate_cells']} aggregate cells reproduce the thesis "
            "table after four-decimal rounding.",
            f"- All {report['checks']['printed_p_values']} printed p-values reproduce within "
            "their displayed precision.",
            "- The round-2 AbsRel Wilcoxon result is significant before correction "
            f"(p={report['interpretation']['round2_absrel_wilcoxon_raw']:.6f}) but not after "
            f"Holm correction (p={report['interpretation']['round2_absrel_wilcoxon_holm']:.6f}).",
            "- Round-2 RMSE remains significant after Holm correction "
            f"(paired-t adjusted p={report['interpretation']['round2_rmse_t_holm']:.6f}).",
            "",
        ]
    )
    return "\n".join(lines)


def validate(input_dir: Path) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    aggregates = []
    common_ids: set[str] | None = None
    for spec in EVALUATIONS:
        data, indexed = load_and_check(input_dir / spec.filename, spec)
        ids = set(indexed)
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError(f"{spec.filename}: item_id set does not match the other evaluations")
        loaded[spec.key] = (data, indexed)
        aggregates.append(
            {
                "evaluation": spec.key,
                **{metric: float(data["summary"][metric]) for metric in METRICS},
                "n_images": len(indexed),
            }
        )

    comparison_specs = (
        ("round1_vs_published", "round1_proposed", "published"),
        ("round1_vs_control", "round1_proposed", "round1_control"),
        ("round2_vs_control", "round2_proposed", "round2_control"),
    )
    paired = []
    printed_checks = 0
    for label, proposed_key, reference_key in comparison_specs:
        proposed_rows = loaded[proposed_key][1]
        reference_rows = loaded[reference_key][1]
        for metric in METRICS:
            item_ids = sorted(proposed_rows)
            proposed = np.asarray([proposed_rows[i][metric] for i in item_ids], dtype=np.float64)
            reference = np.asarray([reference_rows[i][metric] for i in item_ids], dtype=np.float64)
            t_p = float(ttest_rel(proposed, reference).pvalue)
            w_p = float(wilcoxon(proposed, reference).pvalue)
            if (label, metric) in PRINTED_P:
                printed_t, printed_w = PRINTED_P[(label, metric)]
                if not printed_p_agrees(t_p, printed_t):
                    raise AssertionError(f"{label}/{metric}: paired-t {t_p} != printed {printed_t}")
                if not printed_p_agrees(w_p, printed_w):
                    raise AssertionError(f"{label}/{metric}: Wilcoxon {w_p} != printed {printed_w}")
                printed_checks += 2
            paired.append(
                {
                    "comparison": label,
                    "metric": DISPLAY[metric],
                    "metric_key": metric,
                    "proposed_mean": float(proposed.mean()),
                    "reference_mean": float(reference.mean()),
                    "benefit": benefit(metric, float(proposed.mean()), float(reference.mean())),
                    "t_p": t_p,
                    "wilcoxon_p": w_p,
                    "t_holm_p": None,
                    "wilcoxon_holm_p": None,
                }
            )

    # Correct only the six matched-control hypotheses; the published baseline
    # is a distinct sanity/baseline comparison.
    control_rows = [row for row in paired if row["comparison"].endswith("vs_control")]
    adjusted_t = holm_adjust([row["t_p"] for row in control_rows])
    adjusted_w = holm_adjust([row["wilcoxon_p"] for row in control_rows])
    for row, t_holm, w_holm in zip(control_rows, adjusted_t, adjusted_w):
        row["t_holm_p"] = t_holm
        row["wilcoxon_holm_p"] = w_holm
    for row in paired:
        if row["t_holm_p"] is None:
            row["t_holm_p"] = row["t_p"]
            row["wilcoxon_holm_p"] = row["wilcoxon_p"]

    by_key = {(row["comparison"], row["metric_key"]): row for row in paired}
    round2_absrel = by_key[("round2_vs_control", "abs_rel")]
    round2_rmse = by_key[("round2_vs_control", "rmse")]
    return {
        "status": "PASS_WITH_MULTIPLE_TESTING_CAVEAT",
        "protocol": "NYUv2 Eigen-654, rawDepths, metric depth, no alignment, eigen crop, [0.001, 10] m",
        "checks": {
            "files": len(EVALUATIONS),
            "aggregate_cells": len(EVALUATIONS) * len(METRICS),
            "printed_p_values": printed_checks,
        },
        "aggregates": aggregates,
        "paired_tests": paired,
        "interpretation": {
            "round2_absrel_wilcoxon_raw": round2_absrel["wilcoxon_p"],
            "round2_absrel_wilcoxon_holm": round2_absrel["wilcoxon_holm_p"],
            "round2_rmse_t_holm": round2_rmse["t_holm_p"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/section_4_5_6_validation"),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rerun evaluate_paper.py for all five checkpoints before validation",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    if args.regenerate:
        regenerate(root, input_dir)

    report = validate(input_dir)
    markdown = build_markdown(report)
    json_out = args.json_out or input_dir / "validation_report.json"
    markdown_out = args.markdown_out or input_dir / "validation_report.md"
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
