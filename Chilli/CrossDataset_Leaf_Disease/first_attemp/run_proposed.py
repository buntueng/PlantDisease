#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import configure_runtime, load_config
from data_pipeline import build_training_index
from experiment_runner import collect_summary, run_selected_experiments
from proposed_model import build_proposed_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the editable proposed lightweight model under "
                    "the exact same protocol as the baselines."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--experiment",
        choices=["within_cv", "pairwise", "multisource", "pooled_cv", "all"],
        default="within_cv",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional within-CV dataset IDs, e.g. A B C",
    )
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Override config and train the MobileNetV3-derived model from scratch.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    configure_runtime(cfg)

    if args.no_pretrained:
        cfg["training"]["pretrained"] = False

    index_df = build_training_index(cfg, rebuild=args.rebuild_index)
    model_name = str(cfg["proposed"]["name"])
    pretrained = bool(cfg["training"]["pretrained"])

    def builder(num_classes: int):
        return build_proposed_model(
            cfg,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    run_selected_experiments(
        cfg=cfg,
        index_df=index_df,
        family="proposed",
        model_name=model_name,
        model_builder=builder,
        experiment=args.experiment,
        dataset_ids=args.datasets,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        summary = collect_summary(cfg)
        print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
