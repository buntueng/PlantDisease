#!/usr/bin/env python3
"""
Statistical analysis for the chilli cross-dataset experiment suite.

Outputs:
- descriptive_summary.csv
- average_ranks.csv
- friedman_tests.csv
- pairwise_wilcoxon_holm.csv

Optional prediction-level paired tests for two named models:
- mcnemar_exact.csv
- paired_bootstrap_differences.csv

Recommended use:
1) For within-dataset CV:
   python statistical_analysis.py --experiment within_cv --metric test_macro_f1

2) For pairwise cross-dataset experiments after running multiple seeds:
   python statistical_analysis.py --experiment pairwise --metric test_macro_f1

3) For a specific model pair, including exact McNemar tests and paired bootstrap:
   python statistical_analysis.py \
       --experiment pairwise \
       --metric test_macro_f1 \
       --model-a chilli_lite_gfnet \
       --model-b mobilenet_v3_small \
       --bootstrap 2000

Important:
- Wilcoxon tests use matched experimental units only.
- Within/pooled CV units are matched by scenario + fold.
- Pairwise/multisource units are matched by scenario + seed.
- P-values from all pairwise Wilcoxon comparisons within an
  experiment/metric are Holm-adjusted.
- Prediction-level tests align the exact same test images by file path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import (
    binomtest,
    friedmanchisquare,
    rankdata,
    t,
    wilcoxon,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument(
        "--experiment",
        choices=["within_cv", "pairwise", "multisource", "pooled_cv"],
        required=True,
    )
    p.add_argument(
        "--metric",
        default="test_macro_f1",
        help="Run-level metric column from results/summary.csv.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of models for omnibus/pairwise tests.",
    )
    p.add_argument("--model-a", default=None)
    p.add_argument("--model-b", default=None)
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Paired bootstrap replicates for --model-a/--model-b. "
             "Use 2000 or more for final analysis.",
    )
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    return p.parse_args()


def unit_columns(experiment: str) -> list[str]:
    if experiment in {"within_cv", "pooled_cv"}:
        return ["scenario", "fold"]
    return ["scenario", "seed"]


def load_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run at least one experiment first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty.")
    return df


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def rank_biserial_effect(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for x-y."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom > 0 else 0.0


def descriptive_summary(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    rows = []
    for (scenario, model), g in df.groupby(["scenario", "model_name"]):
        values = g[metric].dropna().astype(float).to_numpy()
        n = len(values)
        if n == 0:
            continue

        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        sem = std / math.sqrt(n) if n > 1 else float("nan")
        ci_half = (
            float(t.ppf(0.975, n - 1) * sem)
            if n > 1 and np.isfinite(sem) else float("nan")
        )

        rows.append({
            "scenario": scenario,
            "model_name": model,
            "metric": metric,
            "n": n,
            "mean": mean,
            "std": std,
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "ci95_low": mean - ci_half if np.isfinite(ci_half) else float("nan"),
            "ci95_high": mean + ci_half if np.isfinite(ci_half) else float("nan"),
        })

    return pd.DataFrame(rows)


def build_matched_pivot(
    df: pd.DataFrame,
    experiment: str,
    metric: str,
) -> tuple[pd.DataFrame, list[str]]:
    units = unit_columns(experiment)
    needed = units + ["model_name", metric]
    x = df[needed].dropna(subset=[metric]).copy()

    pivot = x.pivot_table(
        index=units,
        columns="model_name",
        values=metric,
        aggfunc="first",
    )
    return pivot, units


def average_ranks_table(pivot: pd.DataFrame) -> pd.DataFrame:
    complete = pivot.dropna(axis=0, how="any")
    if complete.empty:
        return pd.DataFrame()

    # Higher metric is better, so rank negative values ascending.
    ranks = complete.apply(
        lambda row: pd.Series(
            rankdata(-row.to_numpy(dtype=float), method="average"),
            index=row.index,
        ),
        axis=1,
    )

    out = pd.DataFrame({
        "model_name": ranks.columns,
        "average_rank": ranks.mean(axis=0).to_numpy(),
        "rank_std": ranks.std(axis=0, ddof=1).to_numpy(),
        "n_matched_units": len(ranks),
    }).sort_values("average_rank")

    return out.reset_index(drop=True)


def friedman_test_table(
    pivot: pd.DataFrame,
    experiment: str,
    metric: str,
) -> pd.DataFrame:
    complete = pivot.dropna(axis=0, how="any")

    if complete.shape[0] < 3 or complete.shape[1] < 3:
        return pd.DataFrame([{
            "experiment": experiment,
            "metric": metric,
            "n_matched_units": int(complete.shape[0]),
            "n_models": int(complete.shape[1]),
            "friedman_statistic": float("nan"),
            "p_value": float("nan"),
            "note": "Need >=3 matched units and >=3 models.",
        }])

    arrays = [complete[c].to_numpy(dtype=float) for c in complete.columns]
    stat, p = friedmanchisquare(*arrays)

    return pd.DataFrame([{
        "experiment": experiment,
        "metric": metric,
        "n_matched_units": int(complete.shape[0]),
        "n_models": int(complete.shape[1]),
        "friedman_statistic": float(stat),
        "p_value": float(p),
        "note": "",
    }])


def pairwise_wilcoxon_table(
    pivot: pd.DataFrame,
    experiment: str,
    metric: str,
) -> pd.DataFrame:
    rows = []

    for a, b in itertools.combinations(pivot.columns, 2):
        paired = pivot[[a, b]].dropna()
        x = paired[a].to_numpy(dtype=float)
        y = paired[b].to_numpy(dtype=float)

        if len(x) < 3:
            stat = p = float("nan")
            note = "Fewer than 3 matched units."
        elif np.allclose(x - y, 0):
            stat, p = 0.0, 1.0
            note = "All paired differences are zero."
        else:
            try:
                result = wilcoxon(
                    x,
                    y,
                    zero_method="pratt",
                    alternative="two-sided",
                    correction=False,
                    method="auto",
                )
                stat = float(result.statistic)
                p = float(result.pvalue)
                note = ""
            except ValueError as exc:
                stat = p = float("nan")
                note = str(exc)

        rows.append({
            "experiment": experiment,
            "metric": metric,
            "model_a": a,
            "model_b": b,
            "n_matched_units": len(x),
            "mean_a": float(np.mean(x)) if len(x) else float("nan"),
            "mean_b": float(np.mean(y)) if len(y) else float("nan"),
            "mean_difference_a_minus_b": (
                float(np.mean(x - y)) if len(x) else float("nan")
            ),
            "median_difference_a_minus_b": (
                float(np.median(x - y)) if len(x) else float("nan")
            ),
            "wilcoxon_statistic": stat,
            "p_value_raw": p,
            "rank_biserial_r": (
                rank_biserial_effect(x, y) if len(x) else float("nan")
            ),
            "note": note,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    valid = out["p_value_raw"].notna()
    out["p_value_holm"] = float("nan")
    out.loc[valid, "p_value_holm"] = holm_adjust(
        out.loc[valid, "p_value_raw"].to_numpy()
    )
    out["significant_holm_0_05"] = out["p_value_holm"] < 0.05
    return out


def prediction_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    labels: np.ndarray | None = None,
) -> float:
    if metric_name in {"accuracy", "test_accuracy"}:
        return float(accuracy_score(y_true, y_pred))

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    if metric_name in {"balanced_accuracy", "test_balanced_accuracy"}:
        recalls = []
        for label in labels:
            mask = y_true == label
            if mask.sum() == 0:
                recalls.append(0.0)
            else:
                recalls.append(float(np.mean(y_pred[mask] == label)))
        return float(np.mean(recalls))

    if metric_name in {"macro_f1", "test_macro_f1"}:
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    if metric_name in {"mcc", "test_mcc"}:
        return float(matthews_corrcoef(y_true, y_pred))

    raise ValueError(
        "Prediction-level bootstrap supports accuracy, balanced_accuracy, "
        "macro_f1, and mcc."
    )


def _prediction_path(metrics_path: str) -> Path:
    return Path(metrics_path).resolve().parent / "predictions.csv"


def matched_model_rows(
    df: pd.DataFrame,
    experiment: str,
    model_a: str,
    model_b: str,
) -> list[tuple[dict, dict]]:
    units = unit_columns(experiment)
    a = df[df["model_name"] == model_a].copy()
    b = df[df["model_name"] == model_b].copy()

    merged = a.merge(
        b,
        on=units,
        suffixes=("_a", "_b"),
        how="inner",
    )

    pairs = []
    for _, row in merged.iterrows():
        ra = {c[:-2]: row[c] for c in merged.columns if c.endswith("_a")}
        rb = {c[:-2]: row[c] for c in merged.columns if c.endswith("_b")}
        for u in units:
            ra[u] = row[u]
            rb[u] = row[u]
        pairs.append((ra, rb))

    return pairs


def align_predictions(path_a: Path, path_b: Path) -> pd.DataFrame:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)

    cols = ["path", "y_true_idx", "y_pred_idx"]
    a = a[cols].rename(
        columns={
            "y_true_idx": "y_true_a",
            "y_pred_idx": "y_pred_a",
        }
    )
    b = b[cols].rename(
        columns={
            "y_true_idx": "y_true_b",
            "y_pred_idx": "y_pred_b",
        }
    )

    m = a.merge(b, on="path", how="inner", validate="one_to_one")
    if len(m) == 0:
        raise RuntimeError(
            f"No shared prediction paths between:\n{path_a}\n{path_b}"
        )

    if not np.array_equal(
        m["y_true_a"].to_numpy(),
        m["y_true_b"].to_numpy(),
    ):
        raise RuntimeError(
            "Aligned predictions disagree on y_true indices. "
            "The models may have used different label orders."
        )

    return m


def mcnemar_exact_table(
    df: pd.DataFrame,
    experiment: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    units = unit_columns(experiment)
    rows = []

    for ra, rb in matched_model_rows(df, experiment, model_a, model_b):
        pa = _prediction_path(str(ra["metrics_path"]))
        pb = _prediction_path(str(rb["metrics_path"]))
        m = align_predictions(pa, pb)

        y = m["y_true_a"].to_numpy()
        pred_a = m["y_pred_a"].to_numpy()
        pred_b = m["y_pred_b"].to_numpy()

        correct_a = pred_a == y
        correct_b = pred_b == y

        # b: A correct, B wrong; c: A wrong, B correct
        b_count = int(np.sum(correct_a & ~correct_b))
        c_count = int(np.sum(~correct_a & correct_b))
        discordant = b_count + c_count

        if discordant == 0:
            p_value = 1.0
        else:
            p_value = float(
                binomtest(
                    k=b_count,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        row = {
            "experiment": experiment,
            "model_a": model_a,
            "model_b": model_b,
            "n_aligned_samples": len(m),
            "a_correct_b_wrong": b_count,
            "a_wrong_b_correct": c_count,
            "discordant_total": discordant,
            "mcnemar_exact_p_raw": p_value,
            "accuracy_a": float(correct_a.mean()),
            "accuracy_b": float(correct_b.mean()),
            "accuracy_difference_a_minus_b": float(
                correct_a.mean() - correct_b.mean()
            ),
        }
        for u in units:
            row[u] = ra[u]
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["mcnemar_exact_p_holm"] = holm_adjust(
            out["mcnemar_exact_p_raw"].to_numpy()
        )
        out["significant_holm_0_05"] = (
            out["mcnemar_exact_p_holm"] < 0.05
        )
    return out


def paired_bootstrap_table(
    df: pd.DataFrame,
    experiment: str,
    model_a: str,
    model_b: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()

    units = unit_columns(experiment)
    rows = []
    rng = np.random.default_rng(seed)

    for pair_index, (ra, rb) in enumerate(
        matched_model_rows(df, experiment, model_a, model_b)
    ):
        pa = _prediction_path(str(ra["metrics_path"]))
        pb = _prediction_path(str(rb["metrics_path"]))
        m = align_predictions(pa, pb)

        y = m["y_true_a"].to_numpy(dtype=int)
        pred_a = m["y_pred_a"].to_numpy(dtype=int)
        pred_b = m["y_pred_b"].to_numpy(dtype=int)

        full_labels = np.unique(y)
        observed_a = prediction_metric(y, pred_a, metric, labels=full_labels)
        observed_b = prediction_metric(y, pred_b, metric, labels=full_labels)
        observed_diff = observed_a - observed_b

        diffs = np.empty(n_bootstrap, dtype=float)
        n = len(y)

        # Independent reproducible stream per matched unit.
        local_rng = np.random.default_rng(
            int(rng.integers(0, np.iinfo(np.int32).max))
        )

        for i in range(n_bootstrap):
            idx = local_rng.integers(0, n, size=n)
            ya = y[idx]
            diffs[i] = (
                prediction_metric(
                    ya, pred_a[idx], metric, labels=full_labels
                )
                - prediction_metric(
                    ya, pred_b[idx], metric, labels=full_labels
                )
            )

        row = {
            "experiment": experiment,
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "n_aligned_samples": n,
            "n_bootstrap": n_bootstrap,
            "observed_metric_a": observed_a,
            "observed_metric_b": observed_b,
            "observed_difference_a_minus_b": observed_diff,
            "bootstrap_mean_difference": float(diffs.mean()),
            "ci95_low": float(np.quantile(diffs, 0.025)),
            "ci95_high": float(np.quantile(diffs, 0.975)),
            "bootstrap_probability_a_le_b": float(np.mean(diffs <= 0)),
        }
        for u in units:
            row[u] = ra[u]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "statistics" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(results_dir)
    df = df[df["experiment"] == args.experiment].copy()

    if args.models:
        df = df[df["model_name"].isin(args.models)].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows found for experiment={args.experiment!r}."
        )

    if args.metric not in df.columns:
        raise KeyError(
            f"Metric {args.metric!r} not found. Available test metrics include:\n"
            + "\n".join(sorted(c for c in df.columns if c.startswith("test_")))
        )

    desc = descriptive_summary(df, args.metric)
    desc.to_csv(out_dir / "descriptive_summary.csv", index=False)

    pivot, _ = build_matched_pivot(df, args.experiment, args.metric)

    ranks = average_ranks_table(pivot)
    ranks.to_csv(out_dir / "average_ranks.csv", index=False)

    friedman = friedman_test_table(
        pivot, args.experiment, args.metric
    )
    friedman.to_csv(out_dir / "friedman_tests.csv", index=False)

    wilcox = pairwise_wilcoxon_table(
        pivot, args.experiment, args.metric
    )
    wilcox.to_csv(
        out_dir / "pairwise_wilcoxon_holm.csv",
        index=False,
    )

    print(f"Saved run-level statistical outputs to: {out_dir}")
    print("\nFriedman test:")
    print(friedman.to_string(index=False))

    if not ranks.empty:
        print("\nAverage ranks (lower is better):")
        print(ranks.to_string(index=False))

    if args.model_a or args.model_b:
        if not (args.model_a and args.model_b):
            raise ValueError(
                "Use --model-a and --model-b together."
            )

        pair_df = df[
            df["model_name"].isin([args.model_a, args.model_b])
        ].copy()

        mcnemar = mcnemar_exact_table(
            pair_df,
            args.experiment,
            args.model_a,
            args.model_b,
        )
        mcnemar.to_csv(
            out_dir
            / f"mcnemar_exact__{args.model_a}__vs__{args.model_b}.csv",
            index=False,
        )

        if args.bootstrap > 0:
            boot = paired_bootstrap_table(
                pair_df,
                args.experiment,
                args.model_a,
                args.model_b,
                args.metric,
                args.bootstrap,
                args.bootstrap_seed,
            )
            boot.to_csv(
                out_dir
                / f"paired_bootstrap__{args.model_a}__vs__{args.model_b}.csv",
                index=False,
            )

        print(
            f"\nSaved prediction-level paired tests for "
            f"{args.model_a} vs {args.model_b}."
        )


if __name__ == "__main__":
    main()
