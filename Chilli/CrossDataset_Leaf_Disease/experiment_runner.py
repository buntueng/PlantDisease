from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import canonical_label_order, resolve_path
from data_pipeline import (
    filter_labels,
    labels_present,
    remove_cross_split_exact_overlap,
    shared_labels,
)
from train_engine import fit_and_evaluate


ModelBuilder = Callable[[int], Any]


def _scenario_dir_name(text: str) -> str:
    return (
        text.replace("+", "_plus_")
        .replace("->", "_to_")
        .replace(" ", "_")
    )


def _inner_train_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("inner_val_fraction must be in (0, 0.5).")

    desired_splits = max(2, int(round(1.0 / val_fraction)))
    class_group_counts = df.groupby("label")["group_id"].nunique()
    max_splits = int(class_group_counts.min())
    n_splits = min(desired_splits, max_splits)

    if n_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter.split(
                df,
                y=df["label"],
                groups=df["group_id"],
            )
        )
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
        )

    # Rare fallback for a very small class.
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _save_split_manifest(
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = ["dataset_id", "path", "label", "sha256", "group_id"]
    train_df[cols].to_csv(run_dir / "train_split.csv", index=False)
    val_df[cols].to_csv(run_dir / "val_split.csv", index=False)
    test_df[cols].to_csv(run_dir / "test_split.csv", index=False)


def _validate_split_classes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
) -> None:
    for split_name, frame in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        missing = sorted(set(labels) - set(frame["label"]))
        if missing:
            raise RuntimeError(
                f"{split_name} split is missing classes {missing}. "
                "Inspect class counts and split settings."
            )


def _run_one(
    cfg: dict[str, Any],
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
    experiment: str,
    scenario: str,
    seed: int,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    _validate_split_classes(train_df, val_df, test_df, labels)

    output_root = resolve_path(cfg, cfg["project"]["output_dir"])
    run_dir = (
        output_root
        / family
        / model_name
        / experiment
        / _scenario_dir_name(scenario)
        / f"seed_{seed}"
    )

    if "fold" in extra_metadata:
        run_dir = run_dir / f"fold_{int(extra_metadata['fold']):02d}"

    _save_split_manifest(run_dir, train_df, val_df, test_df)

    model = model_builder(len(labels))
    metadata = {
        "family": family,
        "experiment": experiment,
        "scenario": scenario,
        **extra_metadata,
    }

    print(
        f"\n[{family}] {model_name} | {experiment} | {scenario} | "
        f"seed={seed} | classes={len(labels)}"
    )
    print(
        f"    n_train={len(train_df):,} "
        f"n_val={len(val_df):,} "
        f"n_test={len(test_df):,}"
    )
    print(f"    labels={labels}")

    return fit_and_evaluate(
        model=model,
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_order=labels,
        cfg=cfg,
        run_dir=run_dir,
        seed=seed,
        run_metadata=metadata,
    )


def run_within_cv(
    cfg: dict[str, Any],
    index_df: pd.DataFrame,
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    dataset_ids: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    ecfg = cfg["experiments"]["within_cv"]
    dataset_ids = dataset_ids or list(ecfg["datasets"])
    n_splits = int(ecfg["n_splits"])
    seed = int(ecfg["seed"])
    val_fraction = float(ecfg["inner_val_fraction"])
    results = []

    for dataset_id in dataset_ids:
        df = index_df[index_df["dataset_id"] == dataset_id].reset_index(drop=True)
        labels = labels_present(df, cfg)
        df = filter_labels(df, labels)

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for fold, (pool_idx, test_idx) in enumerate(
            splitter.split(df, y=df["label"], groups=df["group_id"]),
            start=1,
        ):
            pool_df = df.iloc[pool_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            fold_seed = seed + fold - 1
            train_df, val_df = _inner_train_val_split(
                pool_df, val_fraction, fold_seed
            )

            scenario = f"{dataset_id}_within"
            print(
                f"[plan] {model_name} {scenario} fold={fold}/{n_splits} "
                f"classes={labels}"
            )
            if dry_run:
                continue

            results.append(_run_one(
                cfg, family, model_name, model_builder,
                train_df, val_df, test_df, labels,
                experiment="within_cv",
                scenario=scenario,
                seed=fold_seed,
                extra_metadata={
                    "dataset": dataset_id,
                    "fold": fold,
                    "n_splits": n_splits,
                },
            ))

    return results


def run_pairwise(
    cfg: dict[str, Any],
    index_df: pd.DataFrame,
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    ecfg = cfg["experiments"]["pairwise"]
    val_fraction = float(ecfg["inner_val_fraction"])
    results = []

    for source_id, target_id in ecfg["pairs"]:
        source = index_df[index_df["dataset_id"] == source_id].reset_index(drop=True)
        target = index_df[index_df["dataset_id"] == target_id].reset_index(drop=True)

        labels = shared_labels(source, target, cfg)
        if len(labels) < 2:
            raise RuntimeError(
                f"{source_id}->{target_id} has fewer than two shared classes."
            )

        source = filter_labels(source, labels)
        target = filter_labels(target, labels)
        source, target, removed = remove_cross_split_exact_overlap(source, target)

        for seed in [int(x) for x in ecfg["seeds"]]:
            train_df, val_df = _inner_train_val_split(
                source, val_fraction, seed
            )
            scenario = f"{source_id}->{target_id}"

            print(
                f"[plan] {model_name} {scenario} classes={labels} "
                f"exact_target_overlap_removed={removed}"
            )
            if dry_run:
                continue

            results.append(_run_one(
                cfg, family, model_name, model_builder,
                train_df, val_df, target, labels,
                experiment="pairwise",
                scenario=scenario,
                seed=seed,
                extra_metadata={
                    "source_datasets": [source_id],
                    "target_dataset": target_id,
                    "exact_target_overlap_removed": removed,
                },
            ))

    return results


def run_multisource(
    cfg: dict[str, Any],
    index_df: pd.DataFrame,
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    ecfg = cfg["experiments"]["multisource"]
    val_fraction = float(ecfg["inner_val_fraction"])
    results = []

    for spec in ecfg["scenarios"]:
        sources = list(spec["sources"])
        target_id = str(spec["target"])

        source = index_df[index_df["dataset_id"].isin(sources)].reset_index(drop=True)
        target = index_df[index_df["dataset_id"] == target_id].reset_index(drop=True)

        # Multi-source labels are the intersection between:
        # (union of source labels) and target labels.
        labels = shared_labels(source, target, cfg)
        if len(labels) < 2:
            raise RuntimeError(
                f"{sources}->{target_id} has fewer than two testable classes."
            )

        source = filter_labels(source, labels)
        target = filter_labels(target, labels)
        source, target, removed = remove_cross_split_exact_overlap(source, target)

        scenario = f"{'+'.join(sources)}->{target_id}"

        for seed in [int(x) for x in ecfg["seeds"]]:
            train_df, val_df = _inner_train_val_split(
                source, val_fraction, seed
            )

            print(
                f"[plan] {model_name} {scenario} classes={labels} "
                f"exact_target_overlap_removed={removed}"
            )
            if dry_run:
                continue

            results.append(_run_one(
                cfg, family, model_name, model_builder,
                train_df, val_df, target, labels,
                experiment="multisource",
                scenario=scenario,
                seed=seed,
                extra_metadata={
                    "source_datasets": sources,
                    "target_dataset": target_id,
                    "exact_target_overlap_removed": removed,
                },
            ))

    return results


def run_pooled_cv(
    cfg: dict[str, Any],
    index_df: pd.DataFrame,
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    ecfg = cfg["experiments"]["pooled_cv"]
    dataset_ids = list(ecfg["datasets"])
    n_splits = int(ecfg["n_splits"])
    seed = int(ecfg["seed"])
    val_fraction = float(ecfg["inner_val_fraction"])

    df = index_df[index_df["dataset_id"].isin(dataset_ids)].reset_index(drop=True)
    labels = canonical_label_order(cfg, set(df["label"]))
    df = filter_labels(df, labels)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    results = []

    for fold, (pool_idx, test_idx) in enumerate(
        splitter.split(df, y=df["label"], groups=df["group_id"]),
        start=1,
    ):
        pool_df = df.iloc[pool_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        fold_seed = seed + fold - 1

        train_df, val_df = _inner_train_val_split(
            pool_df, val_fraction, fold_seed
        )

        scenario = f"{'+'.join(dataset_ids)}_pooled"
        print(
            f"[plan] {model_name} {scenario} fold={fold}/{n_splits} "
            f"classes={labels}"
        )
        if dry_run:
            continue

        results.append(_run_one(
            cfg, family, model_name, model_builder,
            train_df, val_df, test_df, labels,
            experiment="pooled_cv",
            scenario=scenario,
            seed=fold_seed,
            extra_metadata={
                "datasets": dataset_ids,
                "fold": fold,
                "n_splits": n_splits,
            },
        ))

    return results


def collect_summary(cfg: dict[str, Any]) -> Path:
    output_root = resolve_path(cfg, cfg["project"]["output_dir"])
    rows = []

    if output_root.exists():
        for path in output_root.rglob("metrics.json"):
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            row["metrics_path"] = str(path)
            # Keep labels readable in CSV.
            if isinstance(row.get("label_order"), list):
                row["label_order"] = "|".join(row["label_order"])
            if isinstance(row.get("source_datasets"), list):
                row["source_datasets"] = "|".join(row["source_datasets"])
            if isinstance(row.get("datasets"), list):
                row["datasets"] = "|".join(row["datasets"])
            rows.append(row)

    summary_path = output_root / "summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def run_selected_experiments(
    cfg: dict[str, Any],
    index_df: pd.DataFrame,
    family: str,
    model_name: str,
    model_builder: ModelBuilder,
    experiment: str,
    dataset_ids: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    results = []

    if experiment in ("within_cv", "all"):
        results.extend(run_within_cv(
            cfg, index_df, family, model_name, model_builder,
            dataset_ids=dataset_ids, dry_run=dry_run,
        ))

    if experiment in ("pairwise", "all"):
        results.extend(run_pairwise(
            cfg, index_df, family, model_name, model_builder,
            dry_run=dry_run,
        ))

    if experiment in ("multisource", "all"):
        results.extend(run_multisource(
            cfg, index_df, family, model_name, model_builder,
            dry_run=dry_run,
        ))

    if experiment in ("pooled_cv", "all"):
        results.extend(run_pooled_cv(
            cfg, index_df, family, model_name, model_builder,
            dry_run=dry_run,
        ))

    return results
