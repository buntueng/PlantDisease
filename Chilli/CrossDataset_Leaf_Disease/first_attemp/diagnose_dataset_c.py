#!/usr/bin/env python3
"""
diagnose_dataset_c.py

Diagnose why Dataset C contributes zero usable images.

Run from the project root:

    python3 diagnose_dataset_c.py

Optional:

    python3 diagnose_dataset_c.py --config config.yaml
    python3 diagnose_dataset_c.py --dataset C

Outputs:
    data/manifests/dataset_C_diagnostic.csv
    data/manifests/dataset_C_folder_summary.csv
    data/manifests/dataset_C_status_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}
ARCHIVE_EXTS = {
    ".zip", ".rar", ".7z", ".tar", ".tgz", ".tar.gz",
    ".tar.bz2", ".tar.xz",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", default="C")
    return p.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_any(normalized_path: str, terms: list[str]) -> bool:
    return any(normalize_text(str(term)) in normalized_path for term in terms)


def infer_label(
    normalized_relative_path: str,
    aliases: dict[str, list[str]],
) -> tuple[str | None, list[str]]:
    matches = []
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            alias_n = normalize_text(alias)
            if re.search(
                rf"(^| ){re.escape(alias_n)}($| )",
                normalized_relative_path,
            ):
                matches.append(canonical)
                break

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    dataset_id = args.dataset
    if dataset_id not in cfg["data"]["datasets"]:
        raise KeyError(
            f"Dataset {dataset_id!r} not found in config.yaml. "
            f"Available: {list(cfg['data']['datasets'])}"
        )

    raw_dir = Path(cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = (project_root / raw_dir).resolve()

    ds_cfg = cfg["data"]["datasets"][dataset_id]
    root = raw_dir / ds_cfg["root"]

    manifest_dir = Path(cfg["project"]["scan_log_dir"])
    if not manifest_dir.is_absolute():
        manifest_dir = (project_root / manifest_dir).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    global_exclude = list(cfg["data"].get("exclude_any", []))
    include_any = list(ds_cfg.get("include_any", []))
    local_exclude = list(ds_cfg.get("exclude_any", []))
    aliases = cfg["data"]["label_aliases"]

    print("=" * 80)
    print(f"Dataset diagnostic: {dataset_id}")
    print("=" * 80)
    print(f"Root          : {root}")
    print(f"Exists        : {root.exists()}")
    print(f"include_any   : {include_any}")
    print(f"exclude_any   : {global_exclude + local_exclude}")

    if not root.exists():
        raise FileNotFoundError(root)

    # Show nested archives still present.
    archives = [
        p for p in root.rglob("*")
        if p.is_file()
        and any(p.name.lower().endswith(ext) for ext in ARCHIVE_EXTS)
    ]
    print(f"Nested archives still present: {len(archives)}")
    for p in archives[:30]:
        print(f"  archive: {p.relative_to(root)} ({p.stat().st_size:,} bytes)")

    rows = []
    status_counts = Counter()
    folder_counts = defaultdict(Counter)

    all_images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    print(f"\nAll image-extension files found: {len(all_images):,}")

    for i, path in enumerate(sorted(all_images), start=1):
        rel = path.relative_to(root)
        norm = normalize_text(str(rel))
        size = int(path.stat().st_size)

        included_by_rule = (
            not include_any or contains_any(norm, include_any)
        )
        excluded_by_rule = (
            contains_any(norm, global_exclude)
            or contains_any(norm, local_exclude)
        )

        label, matches = infer_label(norm, aliases)

        width = ""
        height = ""
        image_ok = False
        error = ""
        sha256 = ""

        if size == 0:
            status = "empty_file"
        elif not included_by_rule:
            status = "outside_include_rule"
        elif excluded_by_rule:
            status = "excluded_by_path_rule"
        else:
            try:
                with Image.open(path) as im:
                    width, height = im.size
                    im.verify()
                image_ok = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            if not image_ok:
                status = "corrupt_or_unreadable"
            elif label is None and len(matches) > 1:
                status = "ambiguous_label"
            elif label is None:
                status = "unmatched_label"
            else:
                status = "usable"

        if size > 0:
            try:
                sha256 = sha256_file(path)
            except Exception as exc:
                if not error:
                    error = f"hash_error: {type(exc).__name__}: {exc}"

        status_counts[status] += 1

        # Use the first folder below the dataset root and immediate parent for
        # a practical structure summary.
        parts = rel.parts
        top_folder = parts[0] if len(parts) > 1 else "[root]"
        parent_folder = rel.parent.name if rel.parent != Path(".") else "[root]"
        folder_counts[top_folder][status] += 1

        rows.append({
            "dataset_id": dataset_id,
            "relative_path": str(rel),
            "top_folder": top_folder,
            "parent_folder": parent_folder,
            "size_bytes": size,
            "included_by_rule": included_by_rule,
            "excluded_by_rule": excluded_by_rule,
            "inferred_label": label or "",
            "label_matches": "|".join(matches),
            "width": width,
            "height": height,
            "sha256": sha256,
            "status": status,
            "error": error,
        })

        if i % 1000 == 0:
            print(f"  scanned {i:,}/{len(all_images):,}")

    diagnostic_csv = manifest_dir / f"dataset_{dataset_id}_diagnostic.csv"
    with diagnostic_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "dataset_id", "relative_path", "top_folder", "parent_folder",
            "size_bytes", "included_by_rule", "excluded_by_rule",
            "inferred_label", "label_matches", "width", "height",
            "sha256", "status", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_csv = manifest_dir / f"dataset_{dataset_id}_status_summary.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "count"])
        writer.writeheader()
        for status, count in status_counts.most_common():
            writer.writerow({"status": status, "count": count})

    folder_csv = manifest_dir / f"dataset_{dataset_id}_folder_summary.csv"
    all_statuses = sorted(status_counts)
    with folder_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["top_folder", "total"] + all_statuses
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for folder, counts in sorted(folder_counts.items()):
            row = {
                "top_folder": folder,
                "total": sum(counts.values()),
            }
            row.update({s: counts.get(s, 0) for s in all_statuses})
            writer.writerow(row)

    print("\nStatus summary")
    print("-" * 80)
    for status, count in status_counts.most_common():
        print(f"{status:30s} {count:8,d}")

    print("\nTop parent folders among image files")
    print("-" * 80)
    parent_counter = Counter(r["parent_folder"] for r in rows)
    for folder, count in parent_counter.most_common(30):
        print(f"{folder:40s} {count:8,d}")

    print("\nUsable class counts")
    print("-" * 80)
    usable_labels = Counter(
        r["inferred_label"]
        for r in rows
        if r["status"] == "usable"
    )
    if usable_labels:
        for label, count in usable_labels.most_common():
            print(f"{label:30s} {count:8,d}")
    else:
        print("NO USABLE IMAGES FOUND")

    print("\nWrote:")
    print(f"  {diagnostic_csv}")
    print(f"  {status_csv}")
    print(f"  {folder_csv}")

    print("\nInterpretation")
    print("-" * 80)
    if status_counts["empty_file"] > 0:
        print(
            "* empty_file > 0: the extracted dataset contains zero-byte image "
            "files. Re-extract or verify the source archive."
        )
    if status_counts["corrupt_or_unreadable"] > 0:
        print(
            "* corrupt_or_unreadable > 0: extraction may be incomplete or the "
            "selected representation is damaged."
        )
    if status_counts["outside_include_rule"] > 0:
        print(
            "* outside_include_rule > 0: valid alternatives may exist outside "
            "the configured include_any path."
        )
    if status_counts["unmatched_label"] > 0:
        print(
            "* unmatched_label > 0: add exact Dataset C folder aliases to "
            "config.yaml only after inspecting the folder names."
        )
    if status_counts["usable"] == 0:
        print(
            "* Dataset C currently cannot participate in any experiment. "
            "Do not run pairwise/multisource training until this is resolved."
        )


if __name__ == "__main__":
    main()
