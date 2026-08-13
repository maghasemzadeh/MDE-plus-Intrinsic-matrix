#!/usr/bin/env python3
"""Recompute the historical Section 4.4.2 SILog from fresh saved predictions.

Input must be an output directory produced by ``validate_section_4_4_2.py``.
That validator performs fresh model inference and saves ``arrays.npz`` for
every image. This script reads those newly generated prediction/GT arrays and
applies the January 2026 median-depth-scale SILog protocol used by the thesis.
It never reads the old comparison reports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from tqdm import tqdm


EXPECTED_N = {
    "NYUv2": 1449,
    "DIODE": 769,
    "CityScapes": 2973,
    "DrivingStereo": 2776,
}
DIRECTORIES = {
    "NYUv2": "nyu",
    "DIODE": "diode",
    "CityScapes": "cityscapes",
    "DrivingStereo": "drivingstereo",
}


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=442)
    return parser.parse_args()


def thesis_rows(chapter: Path) -> dict[str, tuple[float, float, float]]:
    text = chapter.read_text(encoding="utf-8")
    table = re.search(
        r"\\begin\{table\}.*?\\label\{tab:da2_revised_comparison\}.*?\\end\{table\}",
        text,
        re.DOTALL,
    )
    if not table:
        raise RuntimeError("section 4.4.2 table not found")
    pattern = re.compile(
        r"(\\lr\{[^}]+\}|[A-Za-z][A-Za-z0-9]*)\s*&\s*"
        r"(\d+(?:\.\d+)?)\s*&\s*(\d+(?:\.\d+)?)\s*&\s*"
        r"(\d+(?:\.\d+)?)\s*&"
    )
    rows = {}
    for match in pattern.finditer(table.group(0)):
        name = re.sub(r"^\\lr\{|\}$", "", match.group(1))
        rows[name] = tuple(float(match.group(i)) for i in (2, 3, 4))
    return rows


def historical_silog(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(ground_truth, dtype=np.float32)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if int(valid.sum()) < 10:
        raise ValueError(f"only {int(valid.sum())} valid pixels")
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    scale1 = np.median(gt_valid) / np.median(pred_valid)
    scaled = (pred_valid * scale1).astype(np.float32)
    scale2 = np.median(gt_valid) / np.median(scaled)
    scaled = scaled * scale2
    log_error = np.log(scaled) - np.log(gt_valid)
    return float(
        np.sqrt(np.mean(log_error**2) - 0.5 * np.mean(log_error) ** 2)
    )


def evaluate_tree(model_root: Path, expected_n: int, label: str) -> list[float]:
    paths = sorted(model_root.rglob("arrays.npz"))
    if len(paths) < expected_n:
        raise RuntimeError(
            f"{label}: expected at least {expected_n} fresh arrays, found "
            f"{len(paths)} in {model_root}"
        )
    values = []
    skipped = []
    for path in tqdm(paths, desc=label, unit="image"):
        try:
            with np.load(path) as arrays:
                values.append(historical_silog(arrays["pred_depth"], arrays["gt_depth"]))
        except ValueError as exc:
            skipped.append(f"{path}: {exc}")
    if len(values) != expected_n:
        raise RuntimeError(
            f"{label}: expected {expected_n} valid arrays, got {len(values)}; "
            f"skipped={skipped[:10]}"
        )
    if skipped:
        print(f"{label}: skipped {len(skipped)} invalid fresh arrays")
        for item in skipped:
            print(f"  - {item}")
    return values


def welch(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    v1 = left.var(ddof=1) / len(left)
    v2 = right.var(ddof=1) / len(right)
    statistic = (left.mean() - right.mean()) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 * v1 / (len(left) - 1) + v2 * v2 / (len(right) - 1))
    p_value = 2.0 * float(student_t.sf(abs(statistic), df))
    return float(statistic), float(df), p_value


def main() -> int:
    args = arguments()
    root = args.repo_root.resolve()
    fresh = args.fresh_output.expanduser().resolve()
    if not fresh.is_dir():
        raise SystemExit(f"fresh output directory not found: {fresh}")
    manifest = fresh / "validation_manifest_4_4_2.json"
    if not manifest.is_file() or not json.loads(manifest.read_text())["fresh_inference"]:
        raise SystemExit("input is not a verified fresh-inference output directory")

    rows = thesis_rows(root / "thesis/tex/chapter4.tex")
    rng = np.random.default_rng(args.seed)
    results = {}
    failures = []
    for thesis_name, directory in DIRECTORIES.items():
        n = EXPECTED_N[thesis_name]
        baseline = np.asarray(
            evaluate_tree(fresh / directory / "da2", n, f"{thesis_name}/baseline")
        )
        proposed = np.asarray(
            evaluate_tree(fresh / directory / "da2-revised", n, f"{thesis_name}/proposed")
        )
        difference = float(baseline.mean() - proposed.mean())
        improvement = difference / float(baseline.mean()) * 100.0
        statistic, df, p_value = welch(baseline, proposed)
        bootstrap = np.empty(args.bootstrap_iterations)
        for index in range(args.bootstrap_iterations):
            bootstrap[index] = (
                rng.choice(baseline, len(baseline), replace=True).mean()
                - rng.choice(proposed, len(proposed), replace=True).mean()
            )
        ci = np.percentile(bootstrap, (2.5, 97.5))
        expected_base, expected_proposed, expected_improvement = rows[thesis_name]
        checks = {
            "baseline_matches": round(float(baseline.mean()), 3) == expected_base,
            "proposed_matches": round(float(proposed.mean()), 3) == expected_proposed,
            "improvement_matches": round(improvement, 2) == expected_improvement,
            "significant": p_value < 0.05,
        }
        for check, passed in checks.items():
            if not passed:
                failures.append(f"{thesis_name}: {check}")
        results[thesis_name] = {
            "baseline_mean": float(baseline.mean()),
            "baseline_std": float(baseline.std(ddof=1)),
            "proposed_mean": float(proposed.mean()),
            "proposed_std": float(proposed.std(ddof=1)),
            "n": n,
            "mean_diff": difference,
            "improvement_percent": improvement,
            "t_statistic": statistic,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "bootstrap_ci": [float(ci[0]), float(ci[1])],
            "thesis": {
                "baseline": expected_base,
                "proposed": expected_proposed,
                "improvement_percent": expected_improvement,
            },
            "checks": checks,
            "per_image_baseline": baseline.tolist(),
            "per_image_proposed": proposed.tolist(),
        }
        print(
            f"{thesis_name}: {baseline.mean():.9f} -> {proposed.mean():.9f}; "
            f"improvement={improvement:.8f}%; p={p_value:.6e}"
        )

    output = args.output
    if output is None:
        output = fresh / "historical_protocol_recomputed_4_4_2.json"
    elif not output.is_absolute():
        output = root / output
    payload = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(fresh),
            "source_fresh_inference": True,
            "protocol": "section-4.4.2-january-2026-median-depth-scale",
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "results": results,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"OUTPUT: {output}")
    print("RESULT: " + (f"FAIL ({len(failures)} mismatch(es))" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
