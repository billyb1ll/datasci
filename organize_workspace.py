#!/usr/bin/env python3
"""
Organize workspace files into tidy folders.

Default behavior is a dry-run (no changes). Use `--apply` to execute moves.

Rules (customizable below):
  - audio/        -> audio files (.mp3, .wav, .flac, .m4a)
  - models/       -> model artifacts (.keras, .h5, .ckpt, .pth, .pt, .onnx)
  - figures/      -> images/plots (.png, .jpg, .jpeg, .svg)
  - notebooks/    -> Jupyter notebooks (.ipynb)
  - logs/         -> logs (.log, nohup.out, tfevents)
  - artifacts/    -> arrays and misc artifacts (.npy, .npz, .pkl, .joblib)
  - data/         -> tabular data (.csv, .parquet) that are in root

Skips existing structured folders by default.
Ensures target directories exist, avoids overwriting by appending numeric suffixes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Folders we consider as already-structured roots and skip walking into from the top-level.
STRUCTURED_DIRS = {
    "audio",
    "models",
    "figures",
    "logs",
    "notebooks",
    "artifacts",
    "data",
    "outputs",
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
}


# Extension to target folder mapping (lowercase extensions only)
EXTENSION_MAP: Dict[str, str] = {
    # audio
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    # images / figures
    ".png": "figures",
    ".jpg": "figures",
    ".jpeg": "figures",
    ".svg": "figures",
    # models
    ".keras": "models",
    ".h5": "models",
    ".ckpt": "models",
    ".pth": "models",
    ".pt": "models",
    ".onnx": "models",
    # artifacts
    ".npy": "artifacts",
    ".npz": "artifacts",
    ".joblib": "artifacts",
    ".pkl": "artifacts",
    # notebooks
    ".ipynb": "notebooks",
    # logs
    ".log": "logs",
    # data (only when in repo root)
    ".csv": "data",
    ".parquet": "data",
}


LOG_BASENAMES = {"nohup.out"}


def is_tensorboard_event(name: str) -> bool:
    return name.startswith("events.out.tfevents")


@dataclass
class MovePlan:
    src: Path
    dst: Path


def next_available_path(dst: Path) -> Path:
    """Return a path that doesn't exist by appending ' (1)', ' (2)', ... before the suffix.
    Example: file.txt -> file (1).txt
    """
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def classify_target(root: Path, path: Path) -> Optional[Path]:
    """Return target directory (absolute Path) for a given file, or None to skip."""
    if not path.is_file():
        return None

    # Skip hidden files in root (like .DS_Store) except known logs
    if path.name.startswith(".") and path.name not in LOG_BASENAMES:
        return None

    # Recognize logs by name
    if path.name in LOG_BASENAMES or is_tensorboard_event(path.name):
        return root / "logs"

    ext = path.suffix.lower()
    target_dir_name = EXTENSION_MAP.get(ext)

    # Classify Python files by name patterns to avoid breaking imports
    if ext == ".py":
        name = path.name.lower()
        stem = path.stem.lower()
        # Scripts/entrypoints
        script_like = (
            name in {"main.py", "main2.py", "main3.py", "evaluate_model.py", "test.py", "data_exploration.py"}
            or stem.startswith("main")
        )
        # Library/util modules
        util_like = (
            "utils" in stem
            or "evaluation" in stem
            or name in {"lr_finder.py"}
        )
        if script_like:
            return root / "scripts"
        if util_like:
            return root / "src"

    if target_dir_name is None:
        return None  # leave unknown types where they are

    # Only move CSV/Parquet in the repository root to avoid touching dataset under data/
    if target_dir_name == "data" and path.parent != root:
        return None

    return root / target_dir_name


def collect_top_level_files(root: Path) -> List[Path]:
    """Collect files directly under repo root that are candidates for moving."""
    files: List[Path] = []
    for child in root.iterdir():
        if child.is_file():
            files.append(child)
        elif child.is_dir():
            # Don't traverse; we only organize items at the top level in this pass
            continue
    return files


def build_plan(root: Path) -> List[MovePlan]:
    plans: List[MovePlan] = []
    for f in collect_top_level_files(root):
        target_dir = classify_target(root, f)
        if target_dir is None:
            continue
        dst = target_dir / f.name
        dst = next_available_path(dst)
        # Do not create self-moves
        if f.resolve() == dst.resolve():
            continue
        plans.append(MovePlan(src=f, dst=dst))
    return plans


def ensure_dirs(plans: Iterable[MovePlan], dry_run: bool = True) -> None:
    dirs = sorted({plan.dst.parent for plan in plans})
    for d in dirs:
        if dry_run:
            print(f"[dry-run] mkdir -p {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)


def execute_moves(plans: Iterable[MovePlan], dry_run: bool = True) -> None:
    for plan in plans:
        rel_src = plan.src.name if plan.src.parent == plan.dst.parent else plan.src.relative_to(plan.src.parents[1] if plan.src.parents else plan.src.parent)
        print(f"{'[dry-run] ' if dry_run else ''}move: {plan.src} -> {plan.dst}")
        if not dry_run:
            plan.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(plan.src), str(plan.dst))


def print_summary(plans: List[MovePlan]) -> None:
    print("\nSummary:")
    by_folder: Dict[str, int] = {}
    for p in plans:
        key = str(p.dst.parent)
        by_folder[key] = by_folder.get(key, 0) + 1
    for folder, count in sorted(by_folder.items()):
        print(f"  {folder}: {count} file(s)")
    print(f"  Total moves: {len(plans)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Organize repository root files into tidy folders.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. By default, runs in dry-run mode and makes no changes.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path.cwd()),
        help="Root directory to organize (default: current working directory)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    # Only act on the top-level; we won't restructure inside known folders in this pass
    plans = build_plan(root)
    if not plans:
        print("Nothing to organize. Repository root looks tidy already.")
        return 0

    print("Planned moves:")
    for p in plans:
        print(f"  {p.src.relative_to(root)} -> {p.dst.relative_to(root)}")

    print_summary(plans)

    dry_run = not args.apply
    ensure_dirs(plans, dry_run=dry_run)
    execute_moves(plans, dry_run=dry_run)

    if dry_run:
        print("\nDry-run complete. Re-run with --apply to perform these moves.")
    else:
        print("\nDone.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())