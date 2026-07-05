#!/usr/bin/env python3
"""
extract_data.py

Extract and organize the three public chilli-leaf datasets downloaded into:

    ./data/zip/

Expected output:

    ./data/
    ├── zip/                         # original downloads; never modified
    ├── raw/
    │   ├── dataset_A_bangladesh/
    │   ├── dataset_B_field8814/
    │   └── dataset_C_cold_india/
    └── manifests/
        ├── extraction_manifest.csv
        ├── image_manifest.csv
        └── dataset_summary.csv

Features
--------
- Maps the three downloaded ZIP files to stable dataset IDs.
- Extracts ZIP/TAR archives safely without flattening paths.
- Recursively extracts nested archives, including nested ZIP files.
- Attempts nested RAR/7z extraction using 7z/7zz/unrar when installed.
- Never deletes or modifies the original downloads.
- Idempotent: completed archives receive marker files and are skipped later.
- Writes manifests with image paths, sizes, parent folders, and SHA-256 hashes.
- Detects exact duplicate image content by SHA-256.
- Produces a compact per-dataset summary.

Run from the project root:

    python extract_data.py

Useful options:

    python extract_data.py --clean
    python extract_data.py --max-depth 10
    python extract_data.py --no-hash

For nested .rar archives on Ubuntu/Debian, install one extractor, e.g.:

    sudo apt install p7zip-full

On Windows, install 7-Zip and ensure 7z.exe is available on PATH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP_DIR = PROJECT_ROOT / "data" / "zip"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

ARCHIVE_EXTS = {
    ".zip", ".rar", ".7z",
    ".tar", ".tgz", ".tbz", ".tbz2", ".txz",
    ".tar.gz", ".tar.bz2", ".tar.xz",
}

# Stable IDs used throughout the research project.
DATASET_A = "dataset_A_bangladesh"
DATASET_B = "dataset_B_field8814"
DATASET_C = "dataset_C_cold_india"


@dataclass(frozen=True)
class ExtractionRecord:
    dataset_id: str
    archive_path: str
    output_dir: str
    archive_type: str
    depth: int
    status: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and organize the three chilli-leaf datasets."
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=DEFAULT_ZIP_DIR,
        help=f"Folder containing downloaded archives (default: {DEFAULT_ZIP_DIR})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Destination for extracted datasets (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Destination for CSV manifests (default: {DEFAULT_MANIFEST_DIR})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum recursive nested-archive extraction depth (default: 8)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete data/raw and data/manifests before extraction.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 hashing for faster manifest generation.",
    )
    return parser.parse_args()


def normalized_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).strip()


def dataset_id_for_archive(path: Path) -> str | None:
    """
    Map top-level downloads to the three planned datasets.

    The matching intentionally tolerates truncated browser filenames.
    """
    name = normalized_name(path)

    # Dataset A
    if (
        "growth stage" in name
        or ("plant leaf disease" in name and "bangladesh" in name)
    ):
        return DATASET_A

    # Dataset B
    if (
        "leaf disease image dataset" in name
        or "classification and early diagnosis" in name
        or ("leaf disease" in name and "classificati" in name)
    ):
        return DATASET_B

    # Dataset C
    if name == "chilli dataset" or name.startswith("chilli dataset "):
        return DATASET_C

    return None


def archive_suffix(path: Path) -> str:
    name = path.name.lower()
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return path.suffix.lower()


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTS)


def archive_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def safe_target(base_dir: Path, member_name: str) -> Path:
    """
    Prevent path traversal such as ../../outside/file.
    """
    base = base_dir.resolve()
    target = (base_dir / member_name).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(
            f"Unsafe archive member path detected: {member_name!r}"
        ) from exc
    return target


def safe_extract_zip(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            # Skip macOS resource-fork metadata.
            if info.filename.startswith("__MACOSX/"):
                continue

            target = safe_target(out_dir, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def safe_extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            safe_target(out_dir, member.name)

        # Avoid special device nodes. Links are skipped to keep the dataset tree
        # self-contained and reduce extraction risk.
        safe_members = [
            m for m in members
            if m.isfile() or m.isdir()
        ]
        tf.extractall(out_dir, members=safe_members, filter="data")


def find_external_extractor() -> tuple[str, str] | None:
    """
    Return (kind, executable_path).

    Detection order
    ---------------
    1. Executables already available on PATH.
    2. Common Windows 7-Zip installation locations.
    3. Common Windows WinRAR installation locations (unrar.exe only).

    This makes the script work when 7-Zip is installed normally on Windows
    but its installation folder was not added to PATH.
    """
    # 1) PATH lookup.
    for exe in ("7zz", "7z", "7z.exe"):
        path = shutil.which(exe)
        if path:
            return "7z", path

    for exe in ("unrar", "unrar.exe"):
        path = shutil.which(exe)
        if path:
            return "unrar", path

    # 2) Common Windows installation locations.
    if os.name == "nt":
        candidates: list[tuple[str, Path]] = []

        env_roots = [
            os.environ.get("ProgramW6432"),
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ]
        for root in env_roots:
            if root:
                candidates.append(
                    ("7z", Path(root) / "7-Zip" / "7z.exe")
                )

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.extend([
                (
                    "7z",
                    Path(local_app_data)
                    / "Programs"
                    / "7-Zip"
                    / "7z.exe",
                ),
                (
                    "7z",
                    Path(local_app_data)
                    / "7-Zip"
                    / "7z.exe",
                ),
            ])

        # WinRAR normally installs UnRAR.exe in the WinRAR folder.
        for root in env_roots:
            if root:
                candidates.append(
                    ("unrar", Path(root) / "WinRAR" / "UnRAR.exe")
                )

        # Literal fallbacks are useful in unusual Python environments where
        # ProgramFiles variables are missing.
        candidates.extend([
            ("7z", Path(r"C:\Program Files\7-Zip\7z.exe")),
            ("7z", Path(r"C:\Program Files (x86)\7-Zip\7z.exe")),
            ("unrar", Path(r"C:\Program Files\WinRAR\UnRAR.exe")),
            ("unrar", Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe")),
        ])

        seen: set[str] = set()
        for kind, candidate in candidates:
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return kind, str(candidate)

    return None


def extract_external(archive: Path, out_dir: Path) -> None:
    extractor = find_external_extractor()
    if extractor is None:
        raise RuntimeError(
            "No RAR/7z extractor found. On Windows, install 7-Zip. "
            "The script checks PATH and common locations such as "
            r"'C:\Program Files\7-Zip\7z.exe'. "
            "On Ubuntu/Debian, install p7zip-full."
        )

    kind, exe = extractor
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "7z":
        cmd = [
            exe, "x",
            str(archive.resolve()),
            f"-o{out_dir.resolve()}",
            "-y",
            "-aos",  # skip existing files
        ]
    else:
        # unrar supports RAR, not generic .7z.
        if archive_suffix(archive) == ".7z":
            raise RuntimeError(
                "A .7z archive was found, but only unrar is installed. "
                "Install 7-Zip/p7zip."
            )
        cmd = [
            exe, "x", "-o+",
            str(archive.resolve()),
            str(out_dir.resolve()) + os.sep,
        ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-25:])
        raise RuntimeError(
            f"External extraction failed with return code {result.returncode}.\n"
            f"Command output tail:\n{tail}"
        )


def extract_one_archive(archive: Path, out_dir: Path) -> str:
    ext = archive_suffix(archive)

    if ext == ".zip":
        safe_extract_zip(archive, out_dir)
        return "zip"

    if ext in {
        ".tar", ".tgz", ".tbz", ".tbz2", ".txz",
        ".tar.gz", ".tar.bz2", ".tar.xz",
    }:
        safe_extract_tar(archive, out_dir)
        return "tar"

    if ext in {".rar", ".7z"}:
        extract_external(archive, out_dir)
        return ext.lstrip(".")

    raise RuntimeError(f"Unsupported archive format: {archive}")


def marker_path(archive: Path) -> Path:
    """
    Marker lives next to the nested archive. For top-level downloads, the caller
    uses a marker inside the dataset root instead.
    """
    digest = hashlib.sha1(str(archive.resolve()).encode("utf-8")).hexdigest()[:12]
    return archive.parent / f".extract_ok_{digest}"


def recursive_extract(
    dataset_id: str,
    root: Path,
    max_depth: int,
    start_depth: int = 1,
) -> list[ExtractionRecord]:
    """
    Repeatedly discover nested archives under `root`.

    Each nested archive is extracted to a sibling directory with the archive's
    stem, e.g.:

        raw/path/cropped.rar
            -> raw/path/cropped/

    This preserves the surrounding hierarchy and avoids flattening.
    """
    records: list[ExtractionRecord] = []

    for depth in range(start_depth, max_depth + 1):
        nested = sorted(
            p for p in root.rglob("*")
            if p.is_file()
            and is_archive(p)
            and "_original_archives" not in p.parts
        )

        work_done = False
        for archive in nested:
            mark = marker_path(archive)
            if mark.exists():
                continue

            out_dir = archive.parent / archive_stem(archive)
            try:
                kind = extract_one_archive(archive, out_dir)
                mark.write_text(
                    f"archive={archive}\nout_dir={out_dir}\n",
                    encoding="utf-8",
                )
                records.append(
                    ExtractionRecord(
                        dataset_id=dataset_id,
                        archive_path=str(archive),
                        output_dir=str(out_dir),
                        archive_type=kind,
                        depth=depth,
                        status="extracted",
                        message="ok",
                    )
                )
                print(f"    [nested:{depth}] OK   {archive} -> {out_dir}")
                work_done = True

            except Exception as exc:
                records.append(
                    ExtractionRecord(
                        dataset_id=dataset_id,
                        archive_path=str(archive),
                        output_dir=str(out_dir),
                        archive_type=archive_suffix(archive).lstrip("."),
                        depth=depth,
                        status="failed",
                        message=str(exc),
                    )
                )
                print(f"    [nested:{depth}] FAIL {archive}")
                print(f"                     {exc}")

                # Mark as attempted only for this run? No. Leave unmarked so that
                # installing 7z and rerunning can succeed.
                continue

        if not work_done:
            # There may still be failed RAR archives; no newly extracted archive
            # means deeper levels cannot appear during this pass.
            break

    return records


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_image_manifest(
    raw_dir: Path,
    manifest_dir: Path,
    do_hash: bool,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    dataset_dirs = [
        p for p in sorted(raw_dir.iterdir())
        if p.is_dir() and p.name.startswith("dataset_")
    ]

    print("\n[scan] Building image manifest...")
    for dataset_dir in dataset_dirs:
        count = 0
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue

            rel = path.relative_to(dataset_dir)
            parent = rel.parent.name if rel.parent != Path(".") else ""
            row = {
                "dataset_id": dataset_dir.name,
                "relative_path": str(rel),
                "parent_folder": parent,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": "" if not do_hash else sha256_file(path),
            }
            rows.append(row)
            count += 1

        print(f"  {dataset_dir.name}: {count:,} images")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_csv = manifest_dir / "image_manifest.csv"
    fieldnames = [
        "dataset_id", "relative_path", "parent_folder",
        "filename", "extension", "size_bytes", "sha256",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write] {out_csv}")
    return rows


def write_extraction_manifest(
    records: Iterable[ExtractionRecord],
    manifest_dir: Path,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_csv = manifest_dir / "extraction_manifest.csv"

    fieldnames = [
        "dataset_id", "archive_path", "output_dir",
        "archive_type", "depth", "status", "message",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "dataset_id": r.dataset_id,
                "archive_path": r.archive_path,
                "output_dir": r.output_dir,
                "archive_type": r.archive_type,
                "depth": r.depth,
                "status": r.status,
                "message": r.message,
            })

    print(f"[write] {out_csv}")


def write_dataset_summary(
    rows: list[dict[str, str | int]],
    manifest_dir: Path,
) -> None:
    out_csv = manifest_dir / "dataset_summary.csv"

    by_dataset: dict[str, list[dict[str, str | int]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset_id"]), []).append(row)

    fieldnames = [
        "dataset_id",
        "image_count",
        "unique_sha256_count",
        "exact_duplicate_extra_files",
        "top_parent_folders",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_id, ds_rows in sorted(by_dataset.items()):
            hashes = [
                str(r["sha256"])
                for r in ds_rows
                if str(r["sha256"])
            ]
            unique_hashes = len(set(hashes)) if hashes else ""
            duplicate_extras = (
                len(hashes) - len(set(hashes))
                if hashes else ""
            )

            parent_counts = Counter(str(r["parent_folder"]) for r in ds_rows)
            top_parent_folders = " | ".join(
                f"{folder or '[root]'}:{count}"
                for folder, count in parent_counts.most_common(20)
            )

            writer.writerow({
                "dataset_id": dataset_id,
                "image_count": len(ds_rows),
                "unique_sha256_count": unique_hashes,
                "exact_duplicate_extra_files": duplicate_extras,
                "top_parent_folders": top_parent_folders,
            })

    print(f"[write] {out_csv}")


def print_duplicate_summary(rows: list[dict[str, str | int]]) -> None:
    hashes = [
        str(r["sha256"])
        for r in rows
        if str(r["sha256"])
    ]
    if not hashes:
        print("\n[duplicates] SHA-256 hashing disabled; exact duplicates not checked.")
        return

    counts = Counter(hashes)
    duplicate_groups = {h: n for h, n in counts.items() if n > 1}
    extra_files = sum(n - 1 for n in duplicate_groups.values())

    print("\n[duplicates]")
    print(f"  Exact duplicate groups : {len(duplicate_groups):,}")
    print(f"  Extra duplicate files  : {extra_files:,}")
    print(
        "  Note: this reports byte-identical images only. "
        "Near-duplicate detection should be a separate preparation step."
    )


def find_top_level_archives(zip_dir: Path) -> list[Path]:
    return sorted(
        p for p in zip_dir.iterdir()
        if p.is_file() and is_archive(p)
    )


def main() -> int:
    args = parse_args()

    zip_dir = args.zip_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()

    print("=" * 78)
    print("Chilli cross-dataset extraction")
    print("=" * 78)
    print(f"Input archives : {zip_dir}")
    print(f"Raw output     : {raw_dir}")
    print(f"Manifests      : {manifest_dir}")
    print(f"Max depth      : {args.max_depth}")
    print(f"SHA-256        : {'off' if args.no_hash else 'on'}")

    if not zip_dir.exists():
        print(f"\nERROR: input folder does not exist: {zip_dir}", file=sys.stderr)
        return 2

    if args.clean:
        print("\n[clean] Removing previous extracted data and manifests...")
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if manifest_dir.exists():
            shutil.rmtree(manifest_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    archives = find_top_level_archives(zip_dir)
    if not archives:
        print(
            f"\nERROR: no supported archives found under {zip_dir}",
            file=sys.stderr,
        )
        return 2

    print("\n[discover] Top-level archives:")
    mapped: dict[str, Path] = {}
    unknown: list[Path] = []

    for archive in archives:
        dataset_id = dataset_id_for_archive(archive)
        if dataset_id is None:
            unknown.append(archive)
            print(f"  ? {archive.name}")
            continue

        if dataset_id in mapped:
            print(
                f"\nERROR: more than one archive mapped to {dataset_id}:\n"
                f"  - {mapped[dataset_id]}\n"
                f"  - {archive}\n"
                "Please keep only the intended top-level download or rename it.",
                file=sys.stderr,
            )
            return 2

        mapped[dataset_id] = archive
        print(f"  ✓ {archive.name}\n      -> {dataset_id}")

    expected = {DATASET_A, DATASET_B, DATASET_C}
    missing = sorted(expected - set(mapped))

    if unknown:
        print("\n[warning] Unmapped archives will be left untouched:")
        for p in unknown:
            print(f"  - {p.name}")

    if missing:
        print("\nERROR: could not find all three expected datasets.", file=sys.stderr)
        print("Missing mappings:", file=sys.stderr)
        for dataset_id in missing:
            print(f"  - {dataset_id}", file=sys.stderr)
        print(
            "\nCheck the filenames in data/zip. The script does not guess "
            "ambiguous dataset identities.",
            file=sys.stderr,
        )
        return 2

    all_records: list[ExtractionRecord] = []

    print("\n[extract] Top-level datasets")
    for dataset_id in (DATASET_A, DATASET_B, DATASET_C):
        archive = mapped[dataset_id]
        dataset_root = raw_dir / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)

        top_marker = dataset_root / ".top_level_extract_ok"
        if top_marker.exists():
            print(f"  [skip] {dataset_id}: already extracted")
        else:
            print(f"  [top]  {archive.name}")
            try:
                kind = extract_one_archive(archive, dataset_root)
                top_marker.write_text(
                    f"archive={archive}\n"
                    f"archive_size={archive.stat().st_size}\n"
                    f"output_dir={dataset_root}\n",
                    encoding="utf-8",
                )
                all_records.append(
                    ExtractionRecord(
                        dataset_id=dataset_id,
                        archive_path=str(archive),
                        output_dir=str(dataset_root),
                        archive_type=kind,
                        depth=0,
                        status="extracted",
                        message="ok",
                    )
                )
                print(f"         -> {dataset_root}")
            except Exception as exc:
                all_records.append(
                    ExtractionRecord(
                        dataset_id=dataset_id,
                        archive_path=str(archive),
                        output_dir=str(dataset_root),
                        archive_type=archive_suffix(archive).lstrip("."),
                        depth=0,
                        status="failed",
                        message=str(exc),
                    )
                )
                print(f"\nERROR extracting {archive}:\n{exc}", file=sys.stderr)
                write_extraction_manifest(all_records, manifest_dir)
                return 1

        print(f"  [scan] nested archives in {dataset_id}")
        nested_records = recursive_extract(
            dataset_id=dataset_id,
            root=dataset_root,
            max_depth=args.max_depth,
            start_depth=1,
        )
        all_records.extend(nested_records)

    print()
    write_extraction_manifest(all_records, manifest_dir)

    rows = build_image_manifest(
        raw_dir=raw_dir,
        manifest_dir=manifest_dir,
        do_hash=not args.no_hash,
    )
    write_dataset_summary(rows, manifest_dir)
    print_duplicate_summary(rows)

    print("\n[done]")
    print("Original downloads were preserved under:")
    print(f"  {zip_dir}")
    print("Extracted datasets are under:")
    for dataset_id in (DATASET_A, DATASET_B, DATASET_C):
        print(f"  {raw_dir / dataset_id}")

    failed = [r for r in all_records if r.status == "failed"]
    if failed:
        print(
            f"\nWARNING: {len(failed)} archive(s) could not be extracted. "
            "See extraction_manifest.csv."
        )
        print(
            "Most commonly this means a nested .rar/.7z archive was found "
            "but 7-Zip/p7zip is not installed."
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
