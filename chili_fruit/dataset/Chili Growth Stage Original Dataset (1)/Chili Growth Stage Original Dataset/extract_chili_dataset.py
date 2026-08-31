#!/usr/bin/env python3
"""Extract or index the five-class Chili Growth Stage dataset.

The script preserves the original dataset distribution. It does NOT create
410 physical images in every class. Training-only balancing is performed by
run_baseline_models.py after each cross-validation split.

Expected canonical classes and original counts:
    Dry Chili       410
    Flower          397
    Green Chili     328
    Red Chili       200
    Rotten Chili    379

Examples:
    python3 extract_chili_dataset.py
    python3 extract_chili_dataset.py --workdir "$PWD" --force
    python3 extract_chili_dataset.py --dataset-root "/path/to/dataset"
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError

SCRIPT_DIR = Path(__file__).resolve().parent

EXPECTED_CLASSES = [
    "Dry Chili",
    "Flower",
    "Green Chili",
    "Red Chili",
    "Rotten Chili",
]
EXPECTED_COUNTS = {
    "Dry Chili": 410,
    "Flower": 397,
    "Green Chili": 328,
    "Red Chili": 200,
    "Rotten Chili": 379,
}


def normalized_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


# Canonical names plus common spelling variants.
CLASS_ALIASES = {
    "Dry Chili": ["Dry Chili", "Dry Chilli", "Dry chili"],
    "Flower": ["Flower", "Flowers"],
    "Green Chili": ["Green Chili", "Green Chilli", "Green chili"],
    "Red Chili": ["Red Chili", "Red Chilli", "Red chili"],
    "Rotten Chili": ["Rotten Chili", "Rotten Chilli", "Rotten chili"],
}
CLASS_LOOKUP = {
    normalized_name(alias): canonical
    for canonical, aliases in CLASS_ALIASES.items()
    for alias in aliases
}


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(destination)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe TAR member path: {member.name}")
        tf.extractall(destination)


def auto_find_archive(workdir: Path) -> Optional[Path]:
    candidates = []
    for pattern in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        candidates.extend(workdir.glob(pattern))
    if not candidates:
        return None
    preferred = [
        path for path in candidates
        if "chili" in path.name.lower()
        and ("growth" in path.name.lower() or "stage" in path.name.lower())
    ]
    return sorted(preferred or candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def extract_archive(archive: Path, destination: Path, force: bool) -> None:
    if destination.exists() and any(destination.iterdir()):
        if not force:
            print(f"[INFO] Extraction skipped because destination is not empty: {destination}")
            return
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if archive.suffix.lower() == ".zip":
        safe_extract_zip(archive, destination)
    elif archive.suffix.lower() == ".tar" or suffixes.endswith((".tar.gz", ".tgz")):
        safe_extract_tar(archive, destination)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")


def direct_class_dirs(parent: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    try:
        children = list(parent.iterdir())
    except (OSError, PermissionError):
        return found
    for child in children:
        if child.is_dir():
            canonical = CLASS_LOOKUP.get(normalized_name(child.name))
            if canonical is not None:
                found[canonical] = child
    return found


def find_dataset_root(search_root: Path) -> Path:
    search_root = search_root.resolve()
    candidates = [search_root]
    candidates.extend(path for path in search_root.rglob("*") if path.is_dir())
    best_root: Optional[Path] = None
    best_count = -1
    for candidate in candidates:
        count = len(direct_class_dirs(candidate))
        if count > best_count:
            best_root, best_count = candidate, count
        if count == len(EXPECTED_CLASSES):
            return candidate
    raise FileNotFoundError(
        f"Could not find all five class folders below {search_root}. "
        f"Best candidate contained {best_count} expected folders: {best_root}\n"
        f"Expected folders: {EXPECTED_CLASSES}"
    )


def iter_candidate_files(folder: Path) -> Iterable[Path]:
    # Pillow decides whether a file is a readable image. This supports JFIF and
    # other valid formats without depending on a hard-coded extension list.
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            yield path


def verify_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, str(exc)


def build_manifest(dataset_root: Path, output_dir: Path, verify: bool) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = direct_class_dirs(dataset_root)
    missing = [name for name in EXPECTED_CLASSES if name not in class_dirs]
    if missing:
        raise FileNotFoundError(f"Missing class folders: {missing}")

    records = []
    invalid_records = []
    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = class_dirs[class_name]
        for image_path in iter_candidate_files(class_dir):
            valid, error = verify_image(image_path) if verify else (True, "")
            if not valid:
                invalid_records.append({
                    "absolute_path": str(image_path.resolve()),
                    "class_name": class_name,
                    "error": error,
                })
                continue
            records.append({
                "image_id": len(records),
                "relative_path": str(image_path.resolve().relative_to(dataset_root.resolve())),
                "absolute_path": str(image_path.resolve()),
                "file_name": image_path.name,
                "class_name": class_name,
                "class_id": class_id,
            })

    invalid_path = output_dir / "invalid_images.csv"
    if invalid_records:
        pd.DataFrame(invalid_records).to_csv(invalid_path, index=False)
    elif invalid_path.exists():
        invalid_path.unlink()

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise RuntimeError("No readable images were found.")

    observed = set(manifest["class_name"].astype(str))
    missing_images = [name for name in EXPECTED_CLASSES if name not in observed]
    if missing_images:
        locations = {name: str(class_dirs[name]) for name in missing_images}
        raise RuntimeError(
            "The following class folders contained no readable images: "
            f"{missing_images}. Folder paths: {locations}. "
            "Check invalid_images.csv and confirm that the class images are not inside another archive."
        )

    manifest = manifest.sort_values(["class_id", "relative_path"]).reset_index(drop=True)
    manifest["image_id"] = range(len(manifest))
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract/index the Chili Growth Stage dataset.")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=SCRIPT_DIR,
        help="Project directory. Default: directory containing this script.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional directory whose direct children are the five class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chili_growth_stage"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image-verification", action="store_true")
    args = parser.parse_args()

    workdir = args.workdir.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (workdir / output_dir).resolve()

    dataset_root = args.dataset_root
    if dataset_root is not None:
        dataset_root = dataset_root.expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (workdir / dataset_root).resolve()
        else:
            dataset_root = dataset_root.resolve()
        print(f"[INFO] Explicit dataset root: {dataset_root}")
    else:
        archive = args.archive
        if archive is not None:
            archive = archive.expanduser()
            if not archive.is_absolute():
                archive = (workdir / archive).resolve()
        else:
            archive = auto_find_archive(workdir)

        if archive is not None:
            print(f"[INFO] Archive: {archive}")
            extract_archive(archive, output_dir, force=args.force)
            search_root = output_dir
        else:
            print("[INFO] No archive detected. Searching for an already-extracted dataset.")
            search_root = workdir
            output_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = find_dataset_root(search_root)

    print(f"[INFO] Dataset root: {dataset_root}")
    manifest = build_manifest(
        dataset_root=dataset_root,
        output_dir=output_dir,
        verify=not args.skip_image_verification,
    )

    counts = manifest["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    info = {
        "dataset_root": str(dataset_root),
        "manifest": str(output_dir / "dataset_manifest.csv"),
        "class_to_id": {name: idx for idx, name in enumerate(EXPECTED_CLASSES)},
        "counts": {name: int(counts[name]) for name in EXPECTED_CLASSES},
        "total_images": int(counts.sum()),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)

    print("\nOriginal images per class:")
    for class_name in EXPECTED_CLASSES:
        actual = int(counts[class_name])
        expected = EXPECTED_COUNTS[class_name]
        status = "OK" if actual == expected else f"CHECK (expected {expected})"
        print(f"  {class_name}: {actual} [{status}]")
    print(f"\nTotal valid original images: {int(counts.sum())}")
    print(f"Manifest: {output_dir / 'dataset_manifest.csv'}")
    print("Training-only balancing to 410 samples/class happens in run_baseline_models.py.")


if __name__ == "__main__":
    main()
