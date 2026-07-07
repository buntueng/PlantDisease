#!/usr/bin/env python3
"""
Merge summary_*.csv files into a single summary.csv and convert
file references from old Ubuntu absolute paths to portable relative paths.

Expected structure:

project_root/
├── baselines/
├── proposed/
├── summary_BLACK.csv
├── summary_ROG.csv
├── summary_SORIN.csv
├── summary_WHITE.csv
└── merge_summaries.py

Output:
├── summary.csv
├── path_conversion_log.csv
└── unresolved_paths.csv          # only created when unresolved paths exist

Usage:
    python3 merge_summaries.py

Windows also works:
    python merge_summaries.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATTERN = "summary_*.csv"
OUTPUT_FILE = "summary.csv"

# Main folders that were copied from Ubuntu machines.
# Add more names here if your project contains other result roots.
KNOWN_ANCHORS = {
    "baselines",
    "proposed",
}

# Column names likely to contain file/directory paths.
PATH_COLUMN_KEYWORDS = {
    "path",
    "file",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "dir",
    "directory",
    "folder",
    "checkpoint",
    "ckpt",
    "model_path",
    "weights",
    "weight_path",
    "history",
    "prediction",
    "predictions",
    "metrics",
    "result",
    "results",
    "output",
    "log",
    "artifact",
    "confusion_matrix",
    "roc",
    "curve",
}

# File extensions that strongly suggest a cell is a path.
PATH_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".log",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".svg",
    ".pdf",
    ".yaml",
    ".yml",
    ".tex",
}


# ============================================================
# CSV handling
# ============================================================

def detect_delimiter(csv_path: Path) -> str:
    """
    Try to detect CSV delimiter.
    Falls back to comma.
    """
    try:
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as f:
            sample = f.read(8192)

        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|"
        )
        return dialect.delimiter

    except (csv.Error, OSError):
        return ","


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    """
    Read CSV using several encoding fallbacks.
    """
    delimiter = detect_delimiter(csv_path)

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                csv_path,
                sep=delimiter,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not read CSV file: {csv_path}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# Path detection
# ============================================================

def normalize_column_name(column: str) -> str:
    """
    Normalize a column name for keyword matching.
    """
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().lower()
    ).strip("_")


def is_path_column(column: str) -> bool:
    """
    Check whether a column name likely stores paths.
    """
    normalized = normalize_column_name(column)

    if normalized in PATH_COLUMN_KEYWORDS:
        return True

    tokens = set(normalized.split("_"))

    return bool(tokens & PATH_COLUMN_KEYWORDS)


def looks_like_path(value: object) -> bool:
    """
    Conservative check for whether a cell looks like a file/directory path.

    This avoids modifying ordinary strings unnecessarily.
    """
    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    # Avoid very long free-text fields.
    if len(text) > 2000:
        return False

    # URL: do not rewrite.
    if re.match(r"^[a-zA-Z]+://", text):
        return False

    # Windows absolute path:
    # C:\folder\file.csv
    # D:/folder/file.csv
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True

    # UNC path:
    # \\server\share\file.csv
    if text.startswith("\\\\"):
        return True

    # Unix absolute path:
    # /home/user/project/file.csv
    if text.startswith("/"):
        return True

    # Home path:
    # ~/project/file.csv
    if text.startswith("~/") or text.startswith("~\\"):
        return True

    # Known copied result directories.
    normalized = text.replace("\\", "/")
    lower_parts = [p.lower() for p in normalized.split("/") if p]

    if any(anchor.lower() in lower_parts for anchor in KNOWN_ANCHORS):
        return True

    # Relative path with known file extension.
    suffix = Path(normalized).suffix.lower()
    if suffix in PATH_EXTENSIONS and "/" in normalized:
        return True

    # Simple filename in a path-oriented column is handled separately.
    return False


# ============================================================
# Path conversion
# ============================================================

def split_path_parts(value: str) -> list[str]:
    """
    Split either Windows or POSIX paths into clean components.
    """
    text = value.strip().replace("\\", "/")

    # Remove Windows drive prefix, e.g. C:
    text = re.sub(r"^[A-Za-z]:", "", text)

    # Remove leading slash(es)
    text = text.lstrip("/")

    return [
        part
        for part in text.split("/")
        if part not in ("", ".")
    ]


def to_portable_relative(path: Path, root: Path) -> str:
    """
    Convert an existing path to a portable relative path using '/'.
    """
    relative = path.resolve().relative_to(root.resolve())
    return relative.as_posix()


def find_from_anchor(
    original_value: str,
    root: Path,
) -> Optional[Path]:
    """
    Map an old path by locating known anchor directories.

    Example:
        /home/user/project/baselines/resnet/run1/file.csv

    becomes:
        ROOT/baselines/resnet/run1/file.csv
    """
    parts = split_path_parts(original_value)

    for i, part in enumerate(parts):
        if part.lower() in {x.lower() for x in KNOWN_ANCHORS}:
            candidate = root.joinpath(*parts[i:])

            if candidate.exists():
                return candidate

    return None


def find_longest_existing_suffix(
    original_value: str,
    root: Path,
) -> Optional[Path]:
    """
    Find the longest path suffix that exists under the new root.

    Example old path:
        /home/machineA/project/results/run3/metrics.csv

    If this exists:
        ROOT/results/run3/metrics.csv

    it will be found.
    """
    parts = split_path_parts(original_value)

    if not parts:
        return None

    # Try increasingly shorter suffixes.
    # Start with the largest possible suffix.
    for start_idx in range(len(parts)):
        suffix_parts = parts[start_idx:]

        if not suffix_parts:
            continue

        candidate = root.joinpath(*suffix_parts)

        if candidate.exists():
            return candidate

    return None


def find_by_basename_unique(
    original_value: str,
    root: Path,
    filename_index: dict[str, list[Path]],
) -> Optional[Path]:
    """
    Locate a file by basename only when exactly one matching file
    exists inside the project.

    This is deliberately conservative.
    """
    parts = split_path_parts(original_value)

    if not parts:
        return None

    basename = parts[-1]

    matches = filename_index.get(basename.lower(), [])

    if len(matches) == 1:
        return matches[0]

    return None


def build_filename_index(
    root: Path,
    ignored_files: set[Path],
) -> dict[str, list[Path]]:
    """
    Build an index:
        filename.lower() -> [matching paths]

    Used as a fallback for moved files.
    """
    index: dict[str, list[Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        try:
            resolved = path.resolve()
        except OSError:
            continue

        if resolved in ignored_files:
            continue

        key = path.name.lower()
        index.setdefault(key, []).append(path)

    return index


def normalize_relative_input(
    original_value: str,
    root: Path,
) -> Optional[Path]:
    """
    Test whether the existing value is already a valid relative path.
    """
    text = original_value.strip().replace("\\", "/")

    # Windows absolute path
    if re.match(r"^[A-Za-z]:/", text):
        return None

    # POSIX absolute path
    if text.startswith("/"):
        return None

    # Home path
    if text.startswith("~/"):
        return None

    candidate = root.joinpath(*PurePosixPath(text).parts)

    if candidate.exists():
        return candidate

    return None


def convert_path_value(
    value: object,
    root: Path,
    filename_index: dict[str, list[Path]],
) -> tuple[str, str]:
    """
    Convert one path value.

    Returns:
        (new_value, conversion_method)

    Methods:
        already_relative
        anchor
        suffix
        unique_basename
        unresolved
        unchanged
    """
    original = str(value).strip()

    if not original:
        return original, "unchanged"

    # Do not modify URLs.
    if re.match(r"^[a-zA-Z]+://", original):
        return original, "unchanged"

    # --------------------------------------------------------
    # 1. Already-correct relative path
    # --------------------------------------------------------
    candidate = normalize_relative_input(original, root)

    if candidate is not None:
        return to_portable_relative(candidate, root), "already_relative"

    # --------------------------------------------------------
    # 2. Match from known anchors: baselines/ or proposed/
    # --------------------------------------------------------
    candidate = find_from_anchor(original, root)

    if candidate is not None:
        return to_portable_relative(candidate, root), "anchor"

    # --------------------------------------------------------
    # 3. Match longest existing suffix
    # --------------------------------------------------------
    candidate = find_longest_existing_suffix(original, root)

    if candidate is not None:
        return to_portable_relative(candidate, root), "suffix"

    # --------------------------------------------------------
    # 4. Match unique filename anywhere under project root
    # --------------------------------------------------------
    candidate = find_by_basename_unique(
        original,
        root,
        filename_index,
    )

    if candidate is not None:
        return to_portable_relative(candidate, root), "unique_basename"

    # --------------------------------------------------------
    # 5. Could not resolve
    # --------------------------------------------------------
    return original, "unresolved"


# ============================================================
# Main merge operation
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge summary_*.csv files and rewrite stale "
            "Ubuntu/Windows paths as project-relative paths."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Project root. Default: directory containing this script."
        ),
    )

    parser.add_argument(
        "--pattern",
        default=INPUT_PATTERN,
        help=f"Input glob pattern. Default: {INPUT_PATTERN}",
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output CSV filename. Default: {OUTPUT_FILE}",
    )

    parser.add_argument(
        "--add-source-column",
        action="store_true",
        help=(
            "Add a '_source_summary' column showing which CSV "
            "each row came from."
        ),
    )

    parser.add_argument(
        "--scan-all-cells",
        action="store_true",
        help=(
            "Also inspect path-looking cells outside columns whose "
            "names indicate paths."
        ),
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output_path = root / args.output
    conversion_log_path = root / "path_conversion_log.csv"
    unresolved_path = root / "unresolved_paths.csv"

    print("=" * 72)
    print("MERGE SUMMARY FILES")
    print("=" * 72)
    print(f"Project root : {root}")
    print(f"Input pattern: {args.pattern}")
    print(f"Output file  : {output_path}")
    print()

    # --------------------------------------------------------
    # Find input CSV files
    # --------------------------------------------------------
    input_files = sorted(root.glob(args.pattern))

    # Prevent accidental inclusion of generated output files.
    excluded_names = {
        output_path.name.lower(),
        conversion_log_path.name.lower(),
        unresolved_path.name.lower(),
    }

    input_files = [
        p for p in input_files
        if p.is_file()
        and p.name.lower() not in excluded_names
    ]

    if not input_files:
        raise FileNotFoundError(
            f"No input CSV files found using:\n"
            f"  {root / args.pattern}"
        )

    print("Input files:")
    for file in input_files:
        print(f"  - {file.name}")
    print()

    # --------------------------------------------------------
    # Read and merge files
    # --------------------------------------------------------
    dataframes: list[pd.DataFrame] = []

    for csv_file in input_files:
        df = read_csv_robust(csv_file)

        if args.add_source_column:
            df["_source_summary"] = csv_file.name

        dataframes.append(df)

        print(
            f"[READ] {csv_file.name:<30} "
            f"rows={len(df):>6}  cols={len(df.columns):>4}"
        )

    merged = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    print()
    print(
        f"Merged dataset: rows={len(merged)}, "
        f"columns={len(merged.columns)}"
    )

    # --------------------------------------------------------
    # Build file index
    # --------------------------------------------------------
    ignored_files = {
        p.resolve() for p in input_files
    } | {
        output_path.resolve(),
        conversion_log_path.resolve(),
        unresolved_path.resolve(),
    }

    print()
    print("Indexing project files...")

    filename_index = build_filename_index(
        root,
        ignored_files,
    )

    print(
        f"Indexed {sum(len(v) for v in filename_index.values())} files."
    )

    # --------------------------------------------------------
    # Identify path columns
    # --------------------------------------------------------
    path_columns = [
        col
        for col in merged.columns
        if is_path_column(str(col))
    ]

    print()
    print("Automatically detected path-related columns:")

    if path_columns:
        for col in path_columns:
            print(f"  - {col}")
    else:
        print("  [none detected by column name]")

    # --------------------------------------------------------
    # Rewrite paths
    # --------------------------------------------------------
    conversion_records: list[dict[str, object]] = []
    unresolved_records: list[dict[str, object]] = []

    changed_count = 0
    unresolved_count = 0

    for row_idx in merged.index:
        for column in merged.columns:
            value = merged.at[row_idx, column]

            if value is None:
                continue

            text = str(value).strip()

            if not text:
                continue

            should_check = column in path_columns

            if args.scan_all_cells and looks_like_path(text):
                should_check = True

            # Within path columns, also support plain file names.
            if not should_check:
                continue

            # Skip numeric-looking values.
            try:
                float(text)
                continue
            except ValueError:
                pass

            new_value, method = convert_path_value(
                text,
                root,
                filename_index,
            )

            if method == "unresolved":
                unresolved_count += 1

                record = {
                    "row_index": row_idx,
                    "column": column,
                    "original_value": text,
                }
                unresolved_records.append(record)

            if new_value != text:
                changed_count += 1

                merged.at[row_idx, column] = new_value

                conversion_records.append({
                    "row_index": row_idx,
                    "column": column,
                    "original_value": text,
                    "new_value": new_value,
                    "method": method,
                })

    # --------------------------------------------------------
    # Save merged CSV
    # --------------------------------------------------------
    merged.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save conversion log.
    pd.DataFrame(
        conversion_records,
        columns=[
            "row_index",
            "column",
            "original_value",
            "new_value",
            "method",
        ],
    ).to_csv(
        conversion_log_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save unresolved paths only when needed.
    if unresolved_records:
        pd.DataFrame(
            unresolved_records,
            columns=[
                "row_index",
                "column",
                "original_value",
            ],
        ).drop_duplicates().to_csv(
            unresolved_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        # Remove stale unresolved report from previous run.
        if unresolved_path.exists():
            unresolved_path.unlink()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"Input CSV files        : {len(input_files)}")
    print(f"Merged rows            : {len(merged)}")
    print(f"Merged columns         : {len(merged.columns)}")
    print(f"Paths changed          : {changed_count}")
    print(f"Unresolved references  : {unresolved_count}")
    print()
    print(f"Created: {output_path}")
    print(f"Created: {conversion_log_path}")

    if unresolved_records:
        print(f"Created: {unresolved_path}")
        print()
        print(
            "WARNING: Some path references could not be mapped "
            "to files under the current project root."
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()