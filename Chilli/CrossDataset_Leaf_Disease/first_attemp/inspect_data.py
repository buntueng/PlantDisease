#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import canonical_label_order, load_config, resolve_path
from data_pipeline import build_training_index, shared_labels


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect harmonized labels before any training."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--rebuild-index", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    df = build_training_index(cfg, rebuild=args.rebuild_index)

    counts = pd.crosstab(df["label"], df["dataset_id"])
    order = canonical_label_order(cfg, set(df["label"]))
    counts = counts.reindex(order).fillna(0).astype(int)

    print("\nHarmonized image counts")
    print(counts.to_string())

    print("\nDataset totals")
    print(df.groupby("dataset_id").size().to_string())

    print("\nPairwise shared classes")
    ids = list(cfg["data"]["datasets"])
    for a in ids:
        for b in ids:
            if a == b:
                continue
            a_df = df[df["dataset_id"] == a]
            b_df = df[df["dataset_id"] == b]
            labels = shared_labels(a_df, b_df, cfg)
            print(f"  {a}->{b}: {len(labels)} classes: {labels}")

    print("\nConfigured multi-source held-out scenarios")
    for spec in cfg["experiments"]["multisource"]["scenarios"]:
        sources = list(spec["sources"])
        target = str(spec["target"])
        src_df = df[df["dataset_id"].isin(sources)]
        tgt_df = df[df["dataset_id"] == target]
        labels = shared_labels(src_df, tgt_df, cfg)
        print(f"  {'+'.join(sources)}->{target}: {len(labels)} classes: {labels}")

    out_dir = resolve_path(cfg, cfg["project"]["scan_log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "dataset_class_matrix.csv")
    print(f"\nSaved: {out_dir / 'dataset_class_matrix.csv'}")


if __name__ == "__main__":
    main()
