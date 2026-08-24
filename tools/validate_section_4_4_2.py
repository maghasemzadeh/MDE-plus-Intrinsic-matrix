#!/usr/bin/env python3
"""Freshly reproduce and validate thesis section 4.4.2.

This script invokes ``compare_models.py`` in a new output directory and runs
model inference again on every raw image. It never treats historical summary
JSON files as fresh validation. Checkpoints are resolved to their current
archival paths and their paths are recorded in the new comparison artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetRun:
    thesis_name: str
    cli_name: str
    baseline_checkpoint: str
    proposed_checkpoint: str


VITL_BASE = "models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth"
VITS_BASE = "models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth"
MULTI_PROPOSED = (
    "models/raw_models/DepthAnythingV2-revised/checkpoints/revised/"
    "best-basic-vitl-multi-database.pth"
)
DRIVING_PROPOSED = (
    # The January artifact records the then-current alias revised/best.pth.
    # That run was archived under its original 2025-12-29 training directory.
    "models/raw_models/DepthAnythingV2-revised/checkpoints/revised/"
    "basic_vits_vkitti_intrinsics_lr5e-6_bs4_20251229_015011/best.pth"
)

RUNS = (
    DatasetRun("NYUv2", "nyu", VITL_BASE, MULTI_PROPOSED),
    DatasetRun("DIODE", "diode", VITL_BASE, MULTI_PROPOSED),
    DatasetRun("CityScapes", "CityScapes", VITL_BASE, MULTI_PROPOSED),
    DatasetRun("DrivingStereo", "DrivingStereo", VITS_BASE, DRIVING_PROPOSED),
)


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument(
        "--max-items",
        type=int,
        help="optional smoke-test cap; omit for the thesis-sized full evaluation",
    )
    return parser.parse_args()


def thesis_rows(chapter: Path) -> dict[str, tuple[float, float, float, str]]:
    text = chapter.read_text(encoding="utf-8")
    table = re.search(
        r"\\begin\{table\}.*?\\label\{tab:da2_revised_comparison\}.*?\\end\{table\}",
        text,
        re.DOTALL,
    )
    if not table:
        raise RuntimeError("section 4.4.2 table was not found")
    pattern = re.compile(
        r"(\\lr\{[^}]+\}|[A-Za-z][A-Za-z0-9]*)\s*&\s*"
        r"\$?(\d+(?:\.\d+)?)\$?\s*&\s*"
        r"\$?(?:\\mathbf\{)?(\d+(?:\.\d+)?)\}?\$?\s*&\s*"
        r"\$?(\d+(?:\.\d+)?)\$?\s*&\s*\$([^$]+)\$\s*\\\\"
    )
    rows = {}
    for match in pattern.finditer(table.group(0)):
        name = re.sub(r"^\\lr\{|\}$", "", match.group(1))
        rows[name] = (
            float(match.group(2)), float(match.group(3)),
            float(match.group(4)), match.group(5).replace(" ", ""),
        )
    return rows


def newest_json(output_dir: Path, before: set[Path]) -> Path:
    created = set(output_dir.glob("*.json")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new summary JSON, found {len(created)}")
    return created.pop()


def main() -> int:
    args = arguments()
    root = args.repo_root.resolve()
    output = args.output_dir
    if output is None:
        output = root / "results" / f"fresh_validation_4_4_2_{datetime.now():%Y%m%d_%H%M%S}"
    elif not output.is_absolute():
        output = root / output
    if output.exists():
        print(f"FAIL: output directory already exists: {output}", file=sys.stderr)
        return 1

    try:
        rows = thesis_rows(root / "thesis/tex/chapter4.tex")
    except (OSError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    missing = []
    for run in RUNS:
        for checkpoint in (run.baseline_checkpoint, run.proposed_checkpoint):
            if not (root / checkpoint).is_file():
                missing.append(f"{run.thesis_name}: {checkpoint}")
    if missing:
        print("FAIL: exact historical checkpoint(s) required for fresh reproduction are missing:")
        for item in missing:
            print(f"  {item}")
        print("No historical report was substituted for missing model inference.")
        return 1

    artifacts: dict[str, Path] = {}
    for run in RUNS:
        before = set(output.glob("*.json")) if output.exists() else set()
        command = [
            sys.executable,
            str(root / "compare_models.py"),
            "--dataset", run.cli_name,
            "--model1", "da2",
            "--model2", "da2-revised",
            "--model1-checkpoint", str(root / run.baseline_checkpoint),
            "--model2-checkpoint", str(root / run.proposed_checkpoint),
            "--model-type", "basic",
            "--outputs-inverse-depth", "auto",
            "--device", args.device,
            "--output-path", str(output),
        ]
        if args.max_items is not None:
            command += ["--max-items", str(args.max_items)]
        print(f"\nFRESH INFERENCE: {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=root, check=True)
            artifacts[run.thesis_name] = newest_json(output, before)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"FAIL: {run.thesis_name}: {exc}", file=sys.stderr)
            return 1

    failures: list[str] = []
    print("\nFresh section 4.4.2 results")
    for run in RUNS:
        payload = json.loads(artifacts[run.thesis_name].read_text(encoding="utf-8"))
        metric = payload["silog"]
        base = float(metric["da2_mean"])
        proposed = float(metric["da2-revised_mean"])
        improvement = (base - proposed) / base * 100.0
        p_value = float(metric["p_value"])
        expected_base, expected_proposed, expected_improvement, _ = rows[run.thesis_name]
        print(
            f"{run.thesis_name}: {base:.9f} -> {proposed:.9f}; "
            f"improvement={improvement:.8f}%; p={p_value:.6e}"
        )
        if args.max_items is None:
            if round(base, 3) != expected_base:
                failures.append(f"{run.thesis_name}: baseline mean mismatch")
            if round(proposed, 3) != expected_proposed:
                failures.append(f"{run.thesis_name}: proposed mean mismatch")
            if round(improvement, 2) != expected_improvement:
                failures.append(
                    f"{run.thesis_name}: thesis improvement {expected_improvement:.2f}%, "
                    f"fresh {improvement:.2f}%"
                )
            if not p_value < 0.05:
                failures.append(f"{run.thesis_name}: significance mismatch")

    manifest = {
        "section": "4.4.2",
        "fresh_inference": True,
        "full_evaluation": args.max_items is None,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "failures": failures,
    }
    manifest_path = output / "validation_manifest_4_4_2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"MANIFEST: {manifest_path}")
    if failures:
        print(f"RESULT: FAIL ({len(failures)} mismatch(es))")
        return 1
    print("RESULT: PASS (fresh inference and evaluation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
