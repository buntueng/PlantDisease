from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common import canonical_label_order, resolve_path


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _contains_any(normalized_path: str, terms: Iterable[str]) -> bool:
    for term in terms:
        if normalize_text(str(term)) in normalized_path:
            return True
    return False


def infer_label(
    normalized_relative_path: str,
    aliases: dict[str, list[str]],
) -> tuple[str | None, list[str]]:
    matches: list[str] = []

    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            alias_n = normalize_text(alias)
            # Word-boundary-style matching after path normalization.
            if re.search(rf"(^| ){re.escape(alias_n)}($| )", normalized_relative_path):
                matches.append(canonical)
                break

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def scan_dataset(
    cfg: dict[str, Any],
    dataset_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Scan one dataset and infer harmonized labels.

    Data-quality safeguards
    -----------------------
    - exclude zero-byte files;
    - optionally verify that PIL can decode each image;
    - exclude images with invalid dimensions;
    - retain an audit log for every excluded item.

    The original files are never modified or deleted.
    """
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    ds_cfg = cfg["data"]["datasets"][dataset_id]
    root = raw_dir / ds_cfg["root"]

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_id} root does not exist: {root}\n"
            "Run extract_data.py first or edit config.yaml."
        )

    global_exclude = cfg["data"].get("exclude_any", [])
    include_any = ds_cfg.get("include_any", [])
    local_exclude = ds_cfg.get("exclude_any", [])
    aliases = cfg["data"]["label_aliases"]
    verify_images = bool(cfg["data"].get("verify_images", True))

    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = path.relative_to(root)
        norm = normalize_text(str(rel))

        if _contains_any(norm, global_exclude) or _contains_any(norm, local_exclude):
            logs.append({
                "dataset_id": dataset_id,
                "path": str(path.resolve()),
                "status": "excluded_by_path_rule",
                "detail": "",
            })
            continue

        if include_any and not _contains_any(norm, include_any):
            logs.append({
                "dataset_id": dataset_id,
                "path": str(path.resolve()),
                "status": "excluded_not_in_include_rule",
                "detail": "",
            })
            continue

        size_bytes = int(path.stat().st_size)
        if size_bytes == 0:
            logs.append({
                "dataset_id": dataset_id,
                "path": str(path.resolve()),
                "status": "empty_file",
                "detail": "size_bytes=0",
            })
            continue

        label, matches = infer_label(norm, aliases)
        if label is None:
            status = "ambiguous_label" if len(matches) > 1 else "unmatched_label"
            logs.append({
                "dataset_id": dataset_id,
                "path": str(path.resolve()),
                "status": status,
                "detail": "|".join(matches),
            })
            continue

        width = height = None
        if verify_images:
            try:
                with Image.open(path) as im:
                    width, height = im.size
                    im.verify()
                if not width or not height or width <= 0 or height <= 0:
                    raise ValueError(
                        f"invalid image dimensions: width={width}, height={height}"
                    )
            except Exception as exc:
                logs.append({
                    "dataset_id": dataset_id,
                    "path": str(path.resolve()),
                    "status": "corrupt_or_unreadable_image",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                continue

        rows.append({
            "dataset_id": dataset_id,
            "path": str(path.resolve()),
            "relative_path": str(rel),
            "label": label,
            "size_bytes": size_bytes,
            "mtime_ns": int(path.stat().st_mtime_ns),
            "width": width,
            "height": height,
        })

    return rows, logs

def build_training_index(
    cfg: dict[str, Any],
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build the harmonized training index with explicit data-quality auditing.

    If byte-identical image content appears with more than one canonical label,
    every member of that SHA-256 group is excluded. The code never guesses the
    correct label and never applies a majority-label rule.
    """
    index_csv = resolve_path(cfg, cfg["project"]["index_csv"])
    log_dir = resolve_path(cfg, cfg["project"]["scan_log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    if index_csv.exists() and not rebuild:
        df = pd.read_csv(index_csv)
        missing = [p for p in df["path"].head(50) if not Path(p).exists()]
        if missing:
            raise RuntimeError(
                "Cached training index points to missing files. "
                "Run with --rebuild-index."
            )
        return df

    all_rows: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []

    for dataset_id in cfg["data"]["datasets"]:
        rows, logs = scan_dataset(cfg, dataset_id)
        all_rows.extend(rows)
        all_logs.extend(logs)

    if not all_rows:
        raise RuntimeError(
            "No labelled images were found. Check data/raw and label aliases "
            "in config.yaml."
        )

    df = pd.DataFrame(all_rows)

    print("[index] Computing SHA-256 hashes for valid images...")
    df["sha256"] = [sha256_file(Path(p)) for p in df["path"]]

    empty_sha256 = hashlib.sha256(b"").hexdigest()
    empty_hash_mask = df["sha256"].eq(empty_sha256)
    if empty_hash_mask.any():
        for _, row in df.loc[empty_hash_mask].iterrows():
            all_logs.append({
                "dataset_id": row["dataset_id"],
                "path": row["path"],
                "status": "empty_content_hash",
                "detail": f"sha256={empty_sha256}",
            })
        df = df.loc[~empty_hash_mask].reset_index(drop=True)

    label_nunique = df.groupby("sha256")["label"].nunique()
    conflict_hashes = set(label_nunique[label_nunique > 1].index)

    if conflict_hashes:
        conflict_df = (
            df[df["sha256"].isin(conflict_hashes)]
            .sort_values(["sha256", "dataset_id", "label", "path"])
            .copy()
        )
        conflict_df["reason"] = "exact_content_has_conflicting_labels"
        conflict_df.to_csv(
            log_dir / "conflicting_duplicate_labels.csv",
            index=False,
        )

        for _, row in conflict_df.iterrows():
            all_logs.append({
                "dataset_id": row["dataset_id"],
                "path": row["path"],
                "status": "excluded_conflicting_duplicate_label",
                "detail": f"sha256={row['sha256']}; label={row['label']}",
            })

        print(
            f"[audit] Excluding {len(conflict_df):,} files from "
            f"{len(conflict_hashes):,} exact-duplicate hash groups with "
            "conflicting labels."
        )
        df = df[~df["sha256"].isin(conflict_hashes)].reset_index(drop=True)
    else:
        pd.DataFrame(columns=[
            "dataset_id", "path", "relative_path", "label",
            "size_bytes", "mtime_ns", "width", "height",
            "sha256", "reason",
        ]).to_csv(
            log_dir / "conflicting_duplicate_labels.csv",
            index=False,
        )

    if df.empty:
        raise RuntimeError(
            "All candidate images were removed during data-quality auditing."
        )

    if bool(cfg["data"].get("drop_exact_duplicates_within_dataset", True)):
        keep_mask = ~df.duplicated(
            subset=["dataset_id", "label", "sha256"],
            keep="first",
        )
        duplicate_removed = df.loc[~keep_mask].copy()

        if not duplicate_removed.empty:
            duplicate_removed["reason"] = (
                "same_label_exact_duplicate_within_dataset"
            )
            duplicate_removed.to_csv(
                log_dir / "removed_exact_duplicates.csv",
                index=False,
            )
            for _, row in duplicate_removed.iterrows():
                all_logs.append({
                    "dataset_id": row["dataset_id"],
                    "path": row["path"],
                    "status": "excluded_same_label_exact_duplicate",
                    "detail": f"sha256={row['sha256']}; label={row['label']}",
                })
        else:
            pd.DataFrame(columns=list(df.columns) + ["reason"]).to_csv(
                log_dir / "removed_exact_duplicates.csv",
                index=False,
            )

        df = df.loc[keep_mask].reset_index(drop=True)

    df["group_id"] = df["sha256"]

    order = {x: i for i, x in enumerate(cfg["data"]["canonical_labels"])}
    df["_label_order"] = df["label"].map(order)
    df = (
        df.sort_values(["dataset_id", "_label_order", "path"])
        .drop(columns="_label_order")
        .reset_index(drop=True)
    )

    index_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_csv, index=False)

    log_df = pd.DataFrame(all_logs)
    if log_df.empty:
        log_df = pd.DataFrame(
            columns=["dataset_id", "path", "status", "detail"]
        )
    log_df.to_csv(log_dir / "scan_log.csv", index=False)

    counts = (
        df.groupby(["dataset_id", "label"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(log_dir / "dataset_class_counts.csv", index=False)

    quality = (
        log_df.groupby(["dataset_id", "status"])
        .size()
        .rename("n")
        .reset_index()
        if not log_df.empty
        else pd.DataFrame(columns=["dataset_id", "status", "n"])
    )
    quality.to_csv(log_dir / "data_quality_summary.csv", index=False)

    print(f"[write] Training index: {index_csv}")
    print(f"[write] Scan log: {log_dir / 'scan_log.csv'}")
    print(
        f"[write] Conflict audit: "
        f"{log_dir / 'conflicting_duplicate_labels.csv'}"
    )
    print(
        f"[write] Data-quality summary: "
        f"{log_dir / 'data_quality_summary.csv'}"
    )

    return df

def remove_cross_split_exact_overlap(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    source_hashes = set(source_df["sha256"])
    overlap_mask = target_df["sha256"].isin(source_hashes)
    removed = int(overlap_mask.sum())
    clean_target = target_df.loc[~overlap_mask].reset_index(drop=True)
    return source_df.reset_index(drop=True), clean_target, removed


def labels_present(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    return canonical_label_order(cfg, set(df["label"]))


def shared_labels(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[str]:
    shared = set(source_df["label"]) & set(target_df["label"])
    return canonical_label_order(cfg, shared)


def filter_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return df[df["label"].isin(labels)].reset_index(drop=True)


def get_transforms(cfg: dict[str, Any]):
    size = int(cfg["data"]["image_size"])
    aug = cfg["augmentation"]["train"]
    jitter = aug["color_jitter"]
    eval_size = int(round(size * float(cfg["augmentation"]["eval_resize_ratio"])))

    # Fixed ImageNet normalization is used by every model.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            size,
            scale=tuple(float(x) for x in aug["random_resized_crop_scale"]),
        ),
        transforms.RandomHorizontalFlip(
            p=float(aug["horizontal_flip_p"])
        ),
        transforms.RandomRotation(
            degrees=float(aug["rotation_degrees"])
        ),
        transforms.ColorJitter(
            brightness=float(jitter["brightness"]),
            contrast=float(jitter["contrast"]),
            saturation=float(jitter["saturation"]),
            hue=float(jitter["hue"]),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(eval_size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, eval_tf


class ImageFrameDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = Path(row["path"])

        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc

        image = self.transform(image)
        target = int(self.class_to_idx[row["label"]])

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "path": str(path),
            "dataset_id": str(row["dataset_id"]),
        }
