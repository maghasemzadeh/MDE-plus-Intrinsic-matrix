#!/usr/bin/env python3
"""Freshly reproduce and validate thesis section 4.4.1.

The default mode loads the released Depth Anything V2 ViT-L basic checkpoint,
runs fresh inference on the four raw datasets, recomputes every per-image SILog
value, and compares the resulting cross-dataset statistics with both the saved
January 2026 artifacts and the thesis table.

Section 4.4.1 predates the repository's February 2026 correction from median
depth scaling to affine inverse-depth alignment.  Reproduction therefore uses
the historical median-scale protocol explicitly.  It does not call the current
``compute_depth_metrics`` implementation, which evaluates a different protocol.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scipy.stats import t as student_t
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required; use the project's Python environment") from exc


SECTION_LABEL = "subsec:basic_model_cross_dataset"
TABLE_LABEL = "tab:basic_model_silog"
EXPECTED_MODEL = "DepthAnythingV2-basic-vitl"
EXPECTED_CHECKPOINT_SHA256 = (
    "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
)
INPUT_SIZE = 518
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 441


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    class_name: str
    expected_n: int
    max_items: int | None = None
    expected_items: int | None = None


@dataclass(frozen=True)
class Comparison:
    thesis_left: str
    thesis_right: str
    left: str
    right: str

    @property
    def name(self) -> str:
        return f"{self.thesis_left} vs {self.thesis_right}"


DATASETS = (
    DatasetSpec("vkitti", "VKITTIDataset", 3000, max_items=3000),
    DatasetSpec("drivingstereo", "DrivingStereoDataset", 2776),
    DatasetSpec("nyu", "NYUDataset", 1449),
    # The raw tree contains 771 pairs. With the experiment's 10 m DIODE cutoff,
    # two outdoor frames have zero valid pixels and are explicitly skipped,
    # yielding the 769 evaluations recorded in the January artifact.
    DatasetSpec("diode", "DIODEDataset", 769, expected_items=771),
)

COMPARISONS = (
    Comparison("VKITTI", "DrivingStereo", "vkitti", "drivingstereo"),
    Comparison("NYUv2", "DrivingStereo", "nyu", "drivingstereo"),
    Comparison("NYUv2", "DIODE", "nyu", "diode"),
)


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--dataset-path", action="append", default=[], metavar="NAME=PATH",
        help="override a dataset root; repeat for any of the four datasets",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact-tolerance", type=float, default=5e-4,
        help="absolute tolerance for fresh vs saved summary values (default: 5e-4)",
    )
    parser.add_argument(
        "--artifacts-only", action="store_true",
        help="perform only the old summary/table consistency check; no inference",
    )
    return parser.parse_args()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def thesis_rows(chapter: Path) -> dict[str, tuple[float, str]]:
    text = chapter.read_text(encoding="utf-8")
    if f"\\label{{{SECTION_LABEL}}}" not in text:
        raise ValueError(f"section label {SECTION_LABEL!r} not found")
    table = re.search(
        rf"\\begin{{table}}.*?\\label{{{TABLE_LABEL}}}.*?\\end{{table}}",
        text, re.DOTALL,
    )
    if not table:
        raise ValueError(f"table {TABLE_LABEL!r} not found")
    pattern = re.compile(
        r"\\lr\{([^}]+)\}\s*&\s*\$?([+-]?\d+(?:\.\d+)?)\$?\s*&\s*\$([^$]+)\$\s*&"
    )
    return {
        match.group(1).strip(): (
            float(match.group(2)), match.group(3).replace(" ", "")
        )
        for match in pattern.finditer(table.group(0))
    }


def artifact_for(root: Path, comparison: Comparison) -> Path:
    pattern = root / "results/base-vitl-vkitti-nyu-diode" / (
        f"depthanythingv2_basic_vitl_{comparison.left}_{comparison.right}_*.json"
    )
    candidates = sorted(Path(value) for value in glob.glob(str(pattern)))
    if not candidates:
        raise FileNotFoundError(f"no saved artifact matches {pattern}")
    return candidates[-1]


def load_saved_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    saved: dict[str, dict[str, Any]] = {}
    for comparison in COMPARISONS:
        path = artifact_for(root, comparison)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric = payload.get("silog", {})
        metadata = payload.get("metadata", {})
        if metadata.get("model_name") != EXPECTED_MODEL:
            raise ValueError(f"{path.name}: unexpected model metadata")
        saved[comparison.name] = {"path": str(path), "metric": metric}
    return saved


def welch_from_summary(
    mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int,
) -> tuple[float, float, float]:
    variance1 = std1 * std1 / n1
    variance2 = std2 * std2 / n2
    standard_error = math.sqrt(variance1 + variance2)
    statistic = (mean1 - mean2) / standard_error
    df = (variance1 + variance2) ** 2 / (
        variance1**2 / (n1 - 1) + variance2**2 / (n2 - 1)
    )
    p_value = 2.0 * float(student_t.sf(abs(statistic), df))
    return statistic, df, p_value


def validate_saved(
    rows: dict[str, tuple[float, str]], saved: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    expected_names = {comparison.name for comparison in COMPARISONS}
    require(set(rows) == expected_names, "thesis table rows are not the expected rows", failures)
    for comparison in COMPARISONS:
        metric = saved[comparison.name]["metric"]
        left_mean = float(metric[f"{comparison.left}_mean"])
        right_mean = float(metric[f"{comparison.right}_mean"])
        left_std = float(metric[f"{comparison.left}_std"])
        right_std = float(metric[f"{comparison.right}_std"])
        left_n = int(metric[f"{comparison.left}_n"])
        right_n = int(metric[f"{comparison.right}_n"])
        difference = float(metric["mean_diff"])
        statistic, _, p_value = welch_from_summary(
            left_mean, left_std, left_n, right_mean, right_std, right_n
        )
        thesis_difference, thesis_p = rows[comparison.name]
        require(math.isclose(difference, left_mean - right_mean, abs_tol=1e-12),
                f"{comparison.name}: saved difference is inconsistent", failures)
        require(math.isclose(statistic, float(metric["t_statistic"]), rel_tol=1e-12),
                f"{comparison.name}: saved t-statistic is inconsistent", failures)
        require(math.isclose(p_value, float(metric["p_value"]), rel_tol=1e-10),
                f"{comparison.name}: saved p-value is inconsistent", failures)
        require(round(abs(difference), 3) == thesis_difference,
                f"{comparison.name}: saved difference disagrees with thesis", failures)
        require(thesis_p == "<0.05" and p_value < 0.05,
                f"{comparison.name}: saved significance disagrees with thesis", failures)
    return failures


def parse_dataset_paths(values: list[str]) -> dict[str, Path]:
    valid = {spec.name for spec in DATASETS}
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset path must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if name not in valid:
            raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(valid)}")
        if name in result:
            raise ValueError(f"duplicate dataset path for {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def article_silog(prediction: Any, ground_truth: Any) -> float:
    """Historical January 2026 per-image SILog implementation."""
    import numpy as np

    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(ground_truth, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != GT shape {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if int(valid.sum()) < 10:
        raise ValueError(f"only {int(valid.sum())} valid pixels")
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    # Preserve both median-scale float32 operations performed by the old pipeline.
    first_scale = np.median(gt_valid) / np.median(pred_valid)
    pred_scaled = (pred_valid * first_scale).astype(np.float32)
    second_scale = np.median(gt_valid) / np.median(pred_scaled)
    pred_scaled = pred_scaled * second_scale
    diff_log = np.log(pred_scaled) - np.log(gt_valid)
    return float(np.sqrt(np.mean(diff_log**2) - 0.5 * np.mean(diff_log) ** 2))


def construct_datasets(overrides: dict[str, Path]) -> list[tuple[DatasetSpec, Any]]:
    from datasets import (
        DIODEDataset, DatasetConfig, DrivingStereoDataset, NYUDataset, VKITTIDataset,
    )

    classes = {
        "DIODEDataset": DIODEDataset,
        "DrivingStereoDataset": DrivingStereoDataset,
        "NYUDataset": NYUDataset,
        "VKITTIDataset": VKITTIDataset,
    }
    result = []
    for spec in DATASETS:
        config = DatasetConfig(
            dataset_path=str(overrides[spec.name]) if spec.name in overrides else None,
            split="train",  # historical CLI default; NYU 'train' means all 1449
            max_items=spec.max_items,
            force_evaluate=True,
        )
        result.append((spec, classes[spec.class_name](config)))
    return result


def preflight(root: Path, overrides: dict[str, Path]) -> None:
    default_roots = {
        "vkitti": root / "datasets/raw_data/vkitti",
        "drivingstereo": root / "datasets/raw_data/DrivingStereo",
        "nyu": root / "datasets/raw_data/nyu",
        "diode": root / "datasets/raw_data/diode",
    }
    markers = {
        "vkitti": ("vkitti_2.0.3_rgb", "vkitti_2.0.3_depth"),
        "drivingstereo": (),
        "nyu": ("nyu_depth_v2_labeled.mat",),
        "diode": ("val",),  # historical train argument falls back to val
    }
    missing: list[str] = []
    for spec in DATASETS:
        dataset_root = overrides.get(spec.name, default_roots[spec.name]).resolve()
        if not dataset_root.is_dir():
            missing.append(f"{spec.name}: missing dataset root {dataset_root}")
            continue
        for marker in markers[spec.name]:
            if not (dataset_root / marker).exists():
                missing.append(
                    f"{spec.name}: missing required path {dataset_root / marker}"
                )
    if missing:
        raise FileNotFoundError("dataset preflight failed:\n  - " + "\n  - ".join(missing))


def evaluate_dataset(spec: DatasetSpec, dataset: Any, model: Any) -> list[float]:
    import numpy as np
    from tqdm import tqdm

    items = dataset.find_items()
    expected_items = spec.expected_items or spec.expected_n
    if len(items) != expected_items:
        raise ValueError(
            f"{spec.name}: expected {expected_items} raw article-protocol items, "
            f"found {len(items)}; refusing to validate a partial/different sample"
        )
    values: list[float] = []
    skipped: list[str] = []
    for item in tqdm(items, desc=f"Fresh inference: {spec.name}", unit="image"):
        try:
            image = dataset.load_image(item)
            gt = np.squeeze(dataset.load_gt_depth(item.gt_path, item))
            intrinsics = dataset.load_intrinsic(item) if dataset.has_intrinsics() else None
            pred = np.squeeze(model.infer_image(image, input_size=INPUT_SIZE, intrinsics=intrinsics))
            values.append(article_silog(pred, gt))
        except Exception as exc:
            skipped.append(f"{item.item_id}: {exc}")
    if len(values) != spec.expected_n:
        details = "\n  - ".join(skipped[:10])
        raise RuntimeError(
            f"{spec.name}: expected {spec.expected_n} valid evaluations, got "
            f"{len(values)} (skipped {len(skipped)})\n  - {details}"
        )
    if skipped:
        print(f"{spec.name}: explicitly skipped {len(skipped)} unusable samples:")
        for detail in skipped:
            print(f"  - {detail}")
    return values


def bootstrap_ci(left: Any, right: Any, rng: Any) -> tuple[float, float]:
    import numpy as np

    differences = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_ITERATIONS):
        differences[index] = (
            rng.choice(left, size=len(left), replace=True).mean()
            - rng.choice(right, size=len(right), replace=True).mean()
        )
    lower, upper = np.percentile(differences, (2.5, 97.5))
    return float(lower), float(upper)


def run_fresh(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    import numpy as np

    # Executing ``python tools/validate_section_4_4_1.py`` puts tools/, rather
    # than the repository root, on sys.path.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    overrides = parse_dataset_paths(args.dataset_path)
    preflight(root, overrides)
    datasets_to_run = construct_datasets(overrides)
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else
        root / "models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"got {checkpoint_hash} ({checkpoint})"
        )
    from models import create_model_wrapper

    model = create_model_wrapper("da2", {
        "model_type": "basic", "encoder": "vitl",
        "checkpoint_path": str(checkpoint), "max_depth": 80.0, "device": args.device,
    })

    per_image: dict[str, list[float]] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for spec, dataset in datasets_to_run:
        values = evaluate_dataset(spec, dataset, model)
        array = np.asarray(values, dtype=np.float64)
        per_image[spec.name] = values
        summaries[spec.name] = {
            "mean": float(array.mean()), "std": float(array.std(ddof=1)), "n": len(values),
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: dict[str, dict[str, float | bool]] = {}
    for comparison in COMPARISONS:
        left = np.asarray(per_image[comparison.left], dtype=np.float64)
        right = np.asarray(per_image[comparison.right], dtype=np.float64)
        left_summary, right_summary = summaries[comparison.left], summaries[comparison.right]
        statistic, df, p_value = welch_from_summary(
            float(left_summary["mean"]), float(left_summary["std"]), int(left_summary["n"]),
            float(right_summary["mean"]), float(right_summary["std"]), int(right_summary["n"]),
        )
        lower, upper = bootstrap_ci(left, right, rng)
        comparisons[comparison.name] = {
            "mean_diff": float(left.mean() - right.mean()), "t_statistic": statistic,
            "degrees_of_freedom": df, "p_value": p_value,
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
            "is_significant": p_value < 0.05,
        }
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fresh_inference": True, "model_name": EXPECTED_MODEL,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "device": args.device or "auto", "input_size": INPUT_SIZE,
            "protocol": "section-4.4.1-january-2026-median-depth-scale",
            "historical_metric_revision": "66bebe5^",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
            "dataset_roots": {spec.name: str(Path(ds.dataset_path).resolve())
                              for spec, ds in datasets_to_run},
        },
        "datasets": summaries, "comparisons": comparisons, "per_image_silog": per_image,
    }


def compare_fresh(
    fresh: dict[str, Any], rows: dict[str, tuple[float, str]],
    saved: dict[str, dict[str, Any]] | None, tolerance: float,
) -> list[str]:
    failures: list[str] = []
    for comparison in COMPARISONS:
        current = fresh["comparisons"][comparison.name]
        difference, p_value = float(current["mean_diff"]), float(current["p_value"])
        thesis_difference, thesis_p = rows[comparison.name]
        require(round(abs(difference), 3) == thesis_difference,
                f"{comparison.name}: fresh |difference| {abs(difference):.9f} does not "
                f"round to thesis value {thesis_difference:.3f}", failures)
        require((p_value < 0.05) == (thesis_p == "<0.05"),
                f"{comparison.name}: fresh p={p_value:.6e} disagrees with thesis {thesis_p}", failures)
        if saved:
            archived = saved[comparison.name]["metric"]
            require(math.isclose(difference, float(archived["mean_diff"]), abs_tol=tolerance,
                                 rel_tol=0.0),
                    f"{comparison.name}: fresh difference {difference:+.9f} differs from saved "
                    f"{float(archived['mean_diff']):+.9f} by more than {tolerance:g}", failures)
            for dataset in (comparison.left, comparison.right):
                fresh_mean = float(fresh["datasets"][dataset]["mean"])
                saved_mean = float(archived[f"{dataset}_mean"])
                require(math.isclose(fresh_mean, saved_mean, abs_tol=tolerance, rel_tol=0.0),
                        f"{comparison.name}: fresh {dataset} mean {fresh_mean:.9f} differs "
                        f"from saved {saved_mean:.9f} by more than {tolerance:g}", failures)
    return failures


def main() -> int:
    args = arguments()
    root = args.repo_root.resolve()
    try:
        rows = thesis_rows(root / "thesis/tex/chapter4.tex")
        saved: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        try:
            saved = load_saved_artifacts(root)
            failures = validate_saved(rows, saved)
        except FileNotFoundError as exc:
            if args.artifacts_only:
                raise
            print(f"WARN: archived summaries unavailable; validating fresh inference "
                  f"directly against the thesis table ({exc})")
        if args.artifacts_only:
            for failure in failures:
                print(f"FAIL: {failure}")
            print("RESULT: " + (f"FAIL ({len(failures)} error(s))" if failures else
                                "PASS (saved artifacts only; inference was not run)"))
            return 1 if failures else 0
        if failures:
            raise ValueError("saved artifact validation failed before fresh inference")

        output = args.output
        if output is None:
            output = root / "results/section-4-4-1-reproduction" / (
                f"fresh_section_4_4_1_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
        elif not output.is_absolute():
            output = root / output
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

        fresh = run_fresh(args, root)
        failures.extend(compare_fresh(fresh, rows, saved, args.artifact_tolerance))
        output.parent.mkdir(parents=True, exist_ok=True)
        fresh["validation"] = {"thesis_rows": rows, "saved_artifacts": saved,
                               "tolerance": args.artifact_tolerance, "failures": failures}
        output.write_text(json.dumps(fresh, indent=2), encoding="utf-8")

        print("\nFresh section 4.4.1 reproduction")
        for comparison in COMPARISONS:
            result = fresh["comparisons"][comparison.name]
            print(f"{comparison.name}: diff={result['mean_diff']:+.9f}; "
                  f"p={result['p_value']:.6e}")
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"REPRODUCTION ARTIFACT: {output}")
        print("RESULT: " + (f"FAIL ({len(failures)} mismatch(es))" if failures else
                            "PASS (fresh inference matches Section 4.4.1)"))
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
