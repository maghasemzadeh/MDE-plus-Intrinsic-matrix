"""Independently reproduce and audit thesis Table 4.9.

The table compares the proposed camera-intrinsics model with (a) the released
Depth Anything V2 model and (b) a matched fine-tuned control, using per-image
paired tests on the 654-image NYUv2 Eigen test split.

Run from the repository root:

    venv/bin/python tools/validate_table_4_9.py

By default, inputs are read from ``results/table_4_9_validation`` and machine-
and human-readable reports are written back to that directory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


METRICS = ("abs_rel", "rmse", "d1")
DISPLAY_NAMES = {"abs_rel": "AbsRel", "rmse": "RMSE", "d1": "delta_1"}
EXPECTED_N = 654
ALPHA = 0.05


@dataclass(frozen=True)
class Comparison:
    encoder: str
    reference: str
    reference_file: str
    proposed_file: str


COMPARISONS = (
    Comparison(
        "ViT-S",
        "published",
        "nyu_basic_vits_crop-eigen_20260704_230541.json",
        "nyu_REVISED_vits_nyuonly_leakfree.json",
    ),
    Comparison(
        "ViT-S",
        "control",
        "nyu_BASEFT_vits_nyuonly_leakfree.json",
        "nyu_REVISED_vits_nyuonly_leakfree.json",
    ),
    Comparison(
        "ViT-L",
        "published",
        "nyu_da2_basic_vitl_crop-eigen_20260709_140905.json",
        "nyu_REVISED_vitl_disp_leakfree.json",
    ),
    Comparison(
        "ViT-L",
        "control",
        "nyu_BASEFT_vitl_disp_leakfree.json",
        "nyu_REVISED_vitl_disp_leakfree.json",
    ),
)


# Numeric values printed in Table 4.9. The validator allows only the rounding
# implied by the table (for example, 0.3961 is correctly printed as 0.40).
EXPECTED_P_VALUES = {
    ("ViT-S", "published", "abs_rel"): (0.40, 0.09),
    ("ViT-S", "published", "rmse"): (2.8e-15, 5.0e-18),
    ("ViT-S", "published", "d1"): (0.34, 0.05),
    ("ViT-S", "control", "abs_rel"): (1.3e-3, 5.0e-3),
    ("ViT-S", "control", "rmse"): (1.1e-5, 3.5e-5),
    ("ViT-S", "control", "d1"): (0.29, 0.60),
    ("ViT-L", "published", "abs_rel"): (2.5e-8, 8.9e-6),
    ("ViT-L", "published", "rmse"): (1.0e-7, 1.1e-9),
    ("ViT-L", "published", "d1"): (4.1e-6, 1.2e-12),
    ("ViT-L", "control", "abs_rel"): (0.77, 0.18),
    ("ViT-L", "control", "rmse"): (0.29, 0.50),
    ("ViT-L", "control", "d1"): (0.81, 0.04),
}


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"summary", "config", "per_image"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level fields {sorted(missing)}")
    return data


def index_per_image(data: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    rows = data["per_image"]
    indexed = {row["item_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{path}: duplicate item_id values")
    if len(indexed) != EXPECTED_N:
        raise ValueError(f"{path}: expected {EXPECTED_N} images, found {len(indexed)}")
    for item_id, row in indexed.items():
        for metric in METRICS:
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{path}: invalid {metric} for {item_id}: {value!r}")
    return indexed


def validate_protocol(data: dict[str, Any], path: Path) -> list[str]:
    config = data["config"]
    expected = {
        "dataset": "nyu",
        "nyu_depth_field": "rawDepths",
        "model_type": "basic",
        "crop": "eigen",
        "min_depth_eval": 0.001,
        "max_depth_eval": 10.0,
        "clip_pred": True,
    }
    checks = []
    for key, wanted in expected.items():
        actual = config.get(key)
        if actual != wanted:
            raise ValueError(f"{path}: config[{key!r}]={actual!r}, expected {wanted!r}")
        checks.append(f"{key}={wanted}")
    return checks


def validate_summary(data: dict[str, Any], indexed: dict[str, dict[str, Any]], path: Path) -> None:
    for metric in METRICS:
        recomputed = float(np.mean([float(row[metric]) for row in indexed.values()]))
        stored = float(data["summary"][metric])
        if not np.isclose(recomputed, stored, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{path}: stored {metric} mean {stored} != per-image mean {recomputed}"
            )
    if int(data["summary"].get("n_images", -1)) != EXPECTED_N:
        raise ValueError(f"{path}: summary n_images is not {EXPECTED_N}")


def agrees_with_printed_value(actual: float, printed: float) -> bool:
    if printed >= 0.01:
        # Decimal entries are displayed to one or two decimal places.
        return abs(actual - printed) <= 0.0050000001
    # Scientific-notation entries have two significant digits (occasionally one).
    return bool(np.isclose(actual, printed, rtol=0.06, atol=0.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise-error adjusted p-values, without statsmodels."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_max = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted.tolist()


def benefit_direction(metric: str, reference_mean: float, proposed_mean: float) -> float:
    """Positive means the proposed model is better."""
    if metric == "d1":
        return proposed_mean - reference_mean
    return reference_mean - proposed_mean


def fmt_p(value: float) -> str:
    return f"{value:.3g}" if value >= 1e-3 else f"{value:.3e}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Table 4.9 independent validation",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "| Encoder | Reference | Metric | Reference mean | Proposed mean | Benefit | Paired t p | Wilcoxon p | Raw conclusion | Holm conclusion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['encoder']} | {row['reference']} | {row['metric']} | "
            f"{row['reference_mean']:.6f} | {row['proposed_mean']:.6f} | "
            f"{row['benefit']:+.6f} | {fmt_p(row['t_p'])} | "
            f"{fmt_p(row['wilcoxon_p'])} | {row['raw_conclusion']} | "
            f"{row['holm_conclusion']} |"
        )
    lines.extend(
        [
            "",
            "`Benefit` is oriented so positive means the proposed model is better. "
            "Holm correction is applied separately to the 12 paired-t and 12 Wilcoxon tests.",
            "",
            f"All {report['checks']['printed_p_values']} printed p-values agree with recomputation within their displayed rounding.",
            f"All {report['checks']['input_files']} input files passed protocol, sample-count, finite-value, ID-pairing, and stored-summary checks.",
            "",
        ]
    )
    return "\n".join(lines)


def validate(input_dir: Path) -> dict[str, Any]:
    cache: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for comparison in COMPARISONS:
        for filename in (comparison.reference_file, comparison.proposed_file):
            if filename in cache:
                continue
            path = input_dir / filename
            data = load_result(path)
            indexed = index_per_image(data, path)
            validate_protocol(data, path)
            validate_summary(data, indexed, path)
            cache[filename] = (data, indexed)

    rows: list[dict[str, Any]] = []
    printed_checks = 0
    for comparison in COMPARISONS:
        reference_data, reference_rows = cache[comparison.reference_file]
        proposed_data, proposed_rows = cache[comparison.proposed_file]
        if set(reference_rows) != set(proposed_rows):
            raise ValueError(
                f"{comparison.encoder}/{comparison.reference}: per-image ID sets do not match"
            )
        item_ids = sorted(reference_rows)
        for metric in METRICS:
            reference_values = np.asarray(
                [reference_rows[item_id][metric] for item_id in item_ids], dtype=np.float64
            )
            proposed_values = np.asarray(
                [proposed_rows[item_id][metric] for item_id in item_ids], dtype=np.float64
            )
            t_p = float(ttest_rel(reference_values, proposed_values).pvalue)
            wilcoxon_p = float(wilcoxon(reference_values, proposed_values).pvalue)
            printed_t, printed_w = EXPECTED_P_VALUES[
                (comparison.encoder, comparison.reference, metric)
            ]
            if not agrees_with_printed_value(t_p, printed_t):
                raise AssertionError(
                    f"paired-t mismatch for {comparison.encoder}/{comparison.reference}/{metric}: "
                    f"actual={t_p}, table={printed_t}"
                )
            if not agrees_with_printed_value(wilcoxon_p, printed_w):
                raise AssertionError(
                    f"Wilcoxon mismatch for {comparison.encoder}/{comparison.reference}/{metric}: "
                    f"actual={wilcoxon_p}, table={printed_w}"
                )
            printed_checks += 2
            reference_mean = float(reference_data["summary"][metric])
            proposed_mean = float(proposed_data["summary"][metric])
            benefit = benefit_direction(metric, reference_mean, proposed_mean)
            rows.append(
                {
                    "encoder": comparison.encoder,
                    "reference": comparison.reference,
                    "metric": DISPLAY_NAMES[metric],
                    "reference_mean": reference_mean,
                    "proposed_mean": proposed_mean,
                    "benefit": benefit,
                    "t_p": t_p,
                    "wilcoxon_p": wilcoxon_p,
                    "table_t_p": printed_t,
                    "table_wilcoxon_p": printed_w,
                }
            )

    adjusted_t = holm_adjust([row["t_p"] for row in rows])
    adjusted_w = holm_adjust([row["wilcoxon_p"] for row in rows])
    for row, t_holm, w_holm in zip(rows, adjusted_t, adjusted_w):
        row["t_holm_p"] = t_holm
        row["wilcoxon_holm_p"] = w_holm
        direction = "proposed better" if row["benefit"] > 0 else "control/reference better"
        raw_sig = row["t_p"] < ALPHA or row["wilcoxon_p"] < ALPHA
        holm_sig = t_holm < ALPHA or w_holm < ALPHA
        row["raw_conclusion"] = direction if raw_sig else "not significant"
        row["holm_conclusion"] = direction if holm_sig else "not significant"

    return {
        "status": "PASS",
        "alpha": ALPHA,
        "correction": "Holm, separately within the 12 paired-t and 12 Wilcoxon tests",
        "checks": {"input_files": len(cache), "printed_p_values": printed_checks},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/table_4_9_validation"),
        help="Directory containing the six archived per-image JSON files",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/table_4_9_validation/validation_report.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("results/table_4_9_validation/validation_report.md"),
    )
    args = parser.parse_args()

    report = validate(args.input_dir)
    markdown = build_markdown(report)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
