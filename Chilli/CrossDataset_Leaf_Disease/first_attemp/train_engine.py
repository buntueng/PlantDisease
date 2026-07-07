from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from common import count_parameters, get_device, save_json, seed_everything
from data_pipeline import ImageFrameDataset, get_transforms


def _make_loader(
    dataset,
    cfg: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    workers = int(cfg["runtime"].get("num_workers", 4))
    persistent = bool(cfg["runtime"].get("persistent_workers", True)) and workers > 0

    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(cfg["runtime"].get("pin_memory", True)),
        persistent_workers=persistent,
        drop_last=False,
    )


def _balanced_class_weights(
    train_df: pd.DataFrame,
    label_order: list[str],
    device: torch.device,
) -> torch.Tensor | None:
    counts = train_df["label"].value_counts()
    n = len(train_df)
    k = len(label_order)
    weights = [n / (k * int(counts[label])) for label in label_order]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _specificity_macro(cm: np.ndarray) -> float:
    total = cm.sum()
    vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        vals.append(float(tn / denom) if denom > 0 else float("nan"))
    return float(np.nanmean(vals))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    labels = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_specificity": _specificity_macro(cm),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    try:
        if n_classes == 2:
            out["auc_ovr_macro"] = float(
                roc_auc_score(y_true, probs[:, 1])
            )
        else:
            out["auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probs,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        out["auc_ovr_macro"] = float("nan")

    return out, cm



def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_order: list[str],
) -> pd.DataFrame:
    """Return one-vs-rest per-class metrics for manuscript tables."""
    n_classes = len(label_order)
    labels = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    rows = []
    total = cm.sum()

    for i, label in enumerate(label_order):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        try:
            auc = float(
                roc_auc_score(
                    (y_true == i).astype(np.int64),
                    probs[:, i],
                )
            )
        except ValueError:
            auc = float("nan")

        rows.append({
            "class_index": i,
            "class_name": label,
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall_sensitivity": float(recall[i]),
            "specificity": specificity,
            "f1": float(f1[i]),
            "auc_ovr": auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return pd.DataFrame(rows)


def split_class_counts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_order: list[str],
) -> pd.DataFrame:
    """Create a stable train/validation/test class-count table."""
    rows = []
    for split_name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        counts = frame["label"].value_counts()
        for label in label_order:
            rows.append({
                "split": split_name,
                "class_name": label,
                "count": int(counts.get(label, 0)),
            })
    return pd.DataFrame(rows)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    grad_clip_norm: float | None,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    prob_rows: list[np.ndarray] = []
    paths: list[str] = []
    dataset_ids: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with _autocast_context(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(grad_clip_norm),
                    )

                scaler.step(optimizer)
                scaler.update()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1)

        losses.append(float(loss.detach().cpu()))
        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        prob_rows.append(probs.cpu().numpy())
        paths.extend(batch["path"])
        dataset_ids.extend(batch["dataset_id"])

    probs_np = np.concatenate(prob_rows, axis=0)
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    metrics, cm = compute_metrics(
        y_true_np,
        y_pred_np,
        probs_np,
        n_classes=probs_np.shape[1],
    )
    metrics["loss"] = float(np.mean(losses))

    return metrics, cm, y_true_np, y_pred_np, probs_np, paths, dataset_ids


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
):
    epochs = int(cfg["training"]["epochs"])
    warmup = int(cfg["training"].get("warmup_epochs", 0))
    base_lr = float(cfg["training"]["learning_rate"])
    min_lr = float(cfg["training"]["min_learning_rate"])
    min_factor = min_lr / base_lr

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)

        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def fit_and_evaluate(
    model: nn.Module,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_order: list[str],
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(
        seed,
        deterministic=bool(cfg["runtime"].get("deterministic", False)),
    )

    device = get_device(cfg)
    model = model.to(device)

    train_tf, eval_tf = get_transforms(cfg)
    class_to_idx = {label: i for i, label in enumerate(label_order)}

    train_ds = ImageFrameDataset(train_df, class_to_idx, train_tf)
    val_ds = ImageFrameDataset(val_df, class_to_idx, eval_tf)
    test_ds = ImageFrameDataset(test_df, class_to_idx, eval_tf)

    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)
    test_loader = _make_loader(test_ds, cfg, shuffle=False)

    class_weights = None
    if str(cfg["training"].get("class_weighting", "none")).lower() == "balanced":
        class_weights = _balanced_class_weights(train_df, label_order, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        betas=tuple(float(x) for x in cfg["training"]["betas"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = _make_scheduler(optimizer, cfg)

    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = _make_scaler(amp_enabled)

    early = cfg["training"]["early_stopping"]
    patience = int(early["patience"])
    min_delta = float(early.get("min_delta", 0.0))
    monitor = str(early.get("monitor", "macro_f1"))
    mode = str(early.get("mode", "max"))

    best_value = -float("inf") if mode == "max" else float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(int(cfg["training"]["epochs"])):
        train_metrics, *_ = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=float(cfg["training"].get("grad_clip_norm", 0.0)),
        )

        with torch.no_grad():
            val_metrics, *_ = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=None,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        current = float(val_metrics[monitor])
        improved = (
            current > best_value + min_delta
            if mode == "max"
            else current < best_value - min_delta
        )

        if improved:
            best_value = current
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"    epoch {epoch+1:03d} "
            f"lr={current_lr:.2e} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"best={best_value:.4f}"
        )

        scheduler.step()

        if epochs_without_improvement >= patience:
            print(f"    early stopping at epoch {epoch+1}")
            break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics, cm, y_true, y_pred, probs, paths, dataset_ids = _run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=None,
        )

    elapsed = time.perf_counter() - start

    peak_gpu_mb = float("nan")
    if device.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    checkpoint_path = run_dir / "best_model.pt"
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "label_order": label_order,
        "seed": seed,
        "best_epoch": best_epoch,
        "config_path": cfg["_config_path"],
        "metadata": run_metadata,
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    pred_df = pd.DataFrame({
        "path": paths,
        "dataset_id": dataset_ids,
        "y_true_idx": y_true,
        "y_pred_idx": y_pred,
        "y_true": [label_order[i] for i in y_true],
        "y_pred": [label_order[i] for i in y_pred],
    })
    for i, label in enumerate(label_order):
        pred_df[f"prob_{label}"] = probs[:, i]
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    cm_df.to_csv(run_dir / "confusion_matrix.csv")

    per_class_df = compute_per_class_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        label_order=label_order,
    )
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)

    split_counts_df = split_class_counts(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_order=label_order,
    )
    split_counts_df.to_csv(run_dir / "split_class_counts.csv", index=False)

    result = {
        **run_metadata,
        "model_name": model_name,
        "seed": int(seed),
        "device": str(device),
        "pretrained": bool(cfg["training"]["pretrained"]),
        "num_classes": len(label_order),
        "label_order": label_order,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "parameters": int(count_parameters(model)),
        "trainable_parameters": int(count_parameters(model, trainable_only=True)),
        "best_epoch": int(best_epoch),
        "best_val_monitor": float(best_value),
        "elapsed_seconds": float(elapsed),
        "peak_gpu_memory_mb": peak_gpu_mb,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    save_json(result, run_dir / "metrics.json")
    save_json(
        {"class_to_idx": class_to_idx, "label_order": label_order},
        run_dir / "labels.json",
    )

    return result
