#!/usr/bin/env python3
"""
repair_dataset_c.py

Repair Dataset C after a partial/broken RAR extraction produced zero-byte images.

What this script does
---------------------
1. Finds Dataset C .rar archives.
2. Locates 7-Zip (7z/7zz), including common Windows locations.
3. Tests each archive with `7z t`.
4. Extracts each archive into a NEW temporary directory using overwrite mode.
5. Validates extracted image files with PIL.
6. Refuses to replace anything if the clean extraction is unusable.
7. Moves the old broken extracted directory to a timestamped backup.
8. Moves the validated clean extraction into place.
9. Writes a CSV repair report.

The original .rar archives are never deleted.

Run from the project root:

    python3 repair_dataset_c.py

Optional:
    python3 repair_dataset_c.py --dataset-root ./data/raw/dataset_C_cold_india
    python3 repair_dataset_c.py --keep-backups
    python3 repair_dataset_c.py --dry-run

After repair:
    python3 inspect_data.py --rebuild-index
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data/raw/dataset_C_cold_india"),
        help="Dataset C root directory.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("./data/manifests"),
        help="Directory for repair report CSV.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be repaired without changing files.",
    )
    p.add_argument(
        "--keep-backups",
        action="store_true",
        help="Keep timestamped backups of broken extracted directories.",
    )
    p.add_argument(
        "--min-valid-images",
        type=int,
        default=10,
        help="Minimum valid images required before replacing a broken folder.",
    )
    return p.parse_args()


def find_7zip() -> str | None:
    for exe in ("7zz", "7z", "7z.exe"):
        found = shutil.which(exe)
        if found:
            return found

    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\7-Zip\7z.exe"),
            Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
        ]

        for env_name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_name)
            if root:
                candidates.append(Path(root) / "7-Zip" / "7z.exe")

        for p in candidates:
            if p.is_file():
                return str(p)

    return None


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def test_archive(seven_zip: str, archive: Path) -> tuple[bool, str]:
    result = run_command([
        seven_zip,
        "t",
        str(archive.resolve()),
        "-y",
    ])
    return result.returncode == 0, result.stdout


def clean_extract(
    seven_zip: str,
    archive: Path,
    temp_out: Path,
) -> tuple[bool, str]:
    """
    Extract to a brand-new temporary directory.

    -aoa = overwrite all existing files.
    In practice the temp directory is new, but -aoa also prevents the
    earlier '-aos skip broken placeholders' problem.
    """
    temp_out.mkdir(parents=True, exist_ok=True)

    result = run_command([
        seven_zip,
        "x",
        str(archive.resolve()),
        f"-o{temp_out.resolve()}",
        "-y",
        "-aoa",
    ])
    return result.returncode == 0, result.stdout


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def validate_images(root: Path) -> dict[str, int]:
    total = 0
    zero_byte = 0
    valid = 0
    corrupt = 0

    for path in image_files(root):
        total += 1
        size = path.stat().st_size

        if size == 0:
            zero_byte += 1
            continue

        try:
            with Image.open(path) as im:
                width, height = im.size
                im.verify()

            if width <= 0 or height <= 0:
                corrupt += 1
            else:
                valid += 1

        except Exception:
            corrupt += 1

    return {
        "total_images": total,
        "valid_images": valid,
        "zero_byte_images": zero_byte,
        "corrupt_images": corrupt,
    }


def infer_existing_target(archive: Path) -> Path:
    """
    Earlier extract_data.py used sibling directory = archive stem.
    Example:
        resized_raw images.rar
          -> resized_raw images/
    """
    return archive.parent / archive.stem


def safe_move(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "archive",
        "archive_size_bytes",
        "archive_test_ok",
        "old_target",
        "old_total_images",
        "old_valid_images",
        "old_zero_byte_images",
        "old_corrupt_images",
        "new_total_images",
        "new_valid_images",
        "new_zero_byte_images",
        "new_corrupt_images",
        "status",
        "message",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_dir = args.report_dir.resolve()

    print("=" * 80)
    print("Dataset C RAR repair")
    print("=" * 80)
    print(f"Dataset root : {dataset_root}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Keep backups : {args.keep_backups}")

    if not dataset_root.exists():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    seven_zip = find_7zip()
    if seven_zip is None:
        print(
            "ERROR: 7-Zip executable not found. Install 7-Zip/p7zip and ensure "
            "`7z` or `7zz` is available.",
            file=sys.stderr,
        )
        return 2

    print(f"7-Zip       : {seven_zip}")

    archives = sorted(dataset_root.rglob("*.rar"))
    if not archives:
        print("ERROR: no .rar archives found.", file=sys.stderr)
        return 2

    print(f"RAR archives : {len(archives)}")
    for p in archives:
        print(f"  - {p.relative_to(dataset_root)} ({p.stat().st_size:,} bytes)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, object]] = []
    failures = 0

    for idx, archive in enumerate(archives, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(archives)}] {archive.name}")

        target = infer_existing_target(archive)
        old_stats = (
            validate_images(target)
            if target.exists()
            else {
                "total_images": 0,
                "valid_images": 0,
                "zero_byte_images": 0,
                "corrupt_images": 0,
            }
        )

        print(f"Old target : {target}")
        print(
            "Old images : "
            f"total={old_stats['total_images']:,}, "
            f"valid={old_stats['valid_images']:,}, "
            f"zero={old_stats['zero_byte_images']:,}, "
            f"corrupt={old_stats['corrupt_images']:,}"
        )

        print("Testing archive...")
        test_ok, test_output = test_archive(seven_zip, archive)

        if not test_ok:
            failures += 1
            tail = "\n".join(test_output.splitlines()[-20:])
            print("FAIL: archive integrity test failed.")
            print(tail)

            rows.append({
                "archive": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_test_ok": False,
                "old_target": str(target),
                "old_total_images": old_stats["total_images"],
                "old_valid_images": old_stats["valid_images"],
                "old_zero_byte_images": old_stats["zero_byte_images"],
                "old_corrupt_images": old_stats["corrupt_images"],
                "new_total_images": 0,
                "new_valid_images": 0,
                "new_zero_byte_images": 0,
                "new_corrupt_images": 0,
                "status": "archive_test_failed",
                "message": tail,
            })
            continue

        print("Archive test: OK")

        if args.dry_run:
            rows.append({
                "archive": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_test_ok": True,
                "old_target": str(target),
                "old_total_images": old_stats["total_images"],
                "old_valid_images": old_stats["valid_images"],
                "old_zero_byte_images": old_stats["zero_byte_images"],
                "old_corrupt_images": old_stats["corrupt_images"],
                "new_total_images": 0,
                "new_valid_images": 0,
                "new_zero_byte_images": 0,
                "new_corrupt_images": 0,
                "status": "dry_run_archive_ok",
                "message": "",
            })
            continue

        temp_parent = archive.parent / f".repair_tmp_{archive.stem}_{timestamp}"
        remove_tree(temp_parent)

        print(f"Clean extraction -> {temp_parent}")
        extract_ok, extract_output = clean_extract(
            seven_zip,
            archive,
            temp_parent,
        )

        if not extract_ok:
            failures += 1
            tail = "\n".join(extract_output.splitlines()[-20:])
            print("FAIL: clean extraction failed.")
            print(tail)
            remove_tree(temp_parent)

            rows.append({
                "archive": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_test_ok": True,
                "old_target": str(target),
                "old_total_images": old_stats["total_images"],
                "old_valid_images": old_stats["valid_images"],
                "old_zero_byte_images": old_stats["zero_byte_images"],
                "old_corrupt_images": old_stats["corrupt_images"],
                "new_total_images": 0,
                "new_valid_images": 0,
                "new_zero_byte_images": 0,
                "new_corrupt_images": 0,
                "status": "clean_extract_failed",
                "message": tail,
            })
            continue

        new_stats = validate_images(temp_parent)
        print(
            "New images : "
            f"total={new_stats['total_images']:,}, "
            f"valid={new_stats['valid_images']:,}, "
            f"zero={new_stats['zero_byte_images']:,}, "
            f"corrupt={new_stats['corrupt_images']:,}"
        )

        # Conservative acceptance criteria.
        acceptable = (
            new_stats["valid_images"] >= args.min_valid_images
            and new_stats["zero_byte_images"] == 0
            and new_stats["corrupt_images"] == 0
        )

        if not acceptable:
            failures += 1
            print(
                "FAIL: clean extraction did not pass validation; "
                "old target left untouched."
            )
            remove_tree(temp_parent)

            rows.append({
                "archive": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_test_ok": True,
                "old_target": str(target),
                "old_total_images": old_stats["total_images"],
                "old_valid_images": old_stats["valid_images"],
                "old_zero_byte_images": old_stats["zero_byte_images"],
                "old_corrupt_images": old_stats["corrupt_images"],
                "new_total_images": new_stats["total_images"],
                "new_valid_images": new_stats["valid_images"],
                "new_zero_byte_images": new_stats["zero_byte_images"],
                "new_corrupt_images": new_stats["corrupt_images"],
                "status": "validation_failed",
                "message": (
                    "Clean extraction failed conservative image validation."
                ),
            })
            continue

        backup = target.parent / f"{target.name}.broken_backup_{timestamp}"

        if target.exists():
            print(f"Backup old target -> {backup}")
            safe_move(target, backup)

        print(f"Install clean target -> {target}")
        safe_move(temp_parent, target)

        # Revalidate final installed directory.
        final_stats = validate_images(target)
        print(
            "Installed   : "
            f"total={final_stats['total_images']:,}, "
            f"valid={final_stats['valid_images']:,}, "
            f"zero={final_stats['zero_byte_images']:,}, "
            f"corrupt={final_stats['corrupt_images']:,}"
        )

        if backup.exists() and not args.keep_backups:
            print(f"Remove broken backup -> {backup}")
            remove_tree(backup)

        rows.append({
            "archive": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_test_ok": True,
            "old_target": str(target),
            "old_total_images": old_stats["total_images"],
            "old_valid_images": old_stats["valid_images"],
            "old_zero_byte_images": old_stats["zero_byte_images"],
            "old_corrupt_images": old_stats["corrupt_images"],
            "new_total_images": final_stats["total_images"],
            "new_valid_images": final_stats["valid_images"],
            "new_zero_byte_images": final_stats["zero_byte_images"],
            "new_corrupt_images": final_stats["corrupt_images"],
            "status": "repaired",
            "message": "",
        })

    report_path = report_dir / "dataset_C_repair_report.csv"
    write_report(rows, report_path)

    print("\n" + "=" * 80)
    print("Repair summary")
    print("=" * 80)
    for row in rows:
        print(
            f"{Path(str(row['archive'])).name:30s} "
            f"{str(row['status']):24s} "
            f"valid={int(row['new_valid_images']):,}"
        )

    print(f"\nReport: {report_path}")

    if failures:
        print(
            f"\nWARNING: {failures} archive(s) were not repaired. "
            "The script left their old targets untouched."
        )
        return 3

    print("\nAll archives repaired successfully.")
    print("\nNext commands:")
    print("  python3 diagnose_dataset_c.py")
    print("  python3 inspect_data.py --rebuild-index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
