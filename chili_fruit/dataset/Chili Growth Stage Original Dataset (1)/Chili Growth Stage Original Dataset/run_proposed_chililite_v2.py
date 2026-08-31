#!/usr/bin/env python3
"""
Train ChiliLite-V2: a lightweight, accuracy-oriented chili growth-stage model.

Design goal
-----------
Keep the parameter count much lower than EfficientNet-B0 while improving the
original MobileNetV3-Small proposal through:

1. ImageNet-pretrained MobileNetV3-Small backbone.
2. Multi-level feature fusion:
   - an intermediate feature map preserves flower edges, chili shape, and texture;
   - the final feature map provides high-level semantic information.
3. Lightweight 3x3/5x5 depthwise refinement and Efficient Channel Attention.
4. Average + learnable GeM pooling for complementary global representations.
5. A small differentiable color-statistics branch for green/red/dry/rotten cues.
6. Color-gated fusion that recalibrates image features with almost no overhead.

Fair-comparison protocol
------------------------
This script uses the same leakage-safe stratified cross-validation engine AND
the same training protocol as run_baseline_models.py. Every training
hyperparameter is inherited from ExperimentConfig, which is the single source
of truth shared with the baselines. Nothing is redefined here, so the two
scripts cannot silently drift apart.

Only the architecture differs between the proposed model and the baselines.
That is the whole point: if ChiliLite-V2 wins, the win is attributable to the
architecture and not to a longer schedule or a retuned optimizer.

A startup audit compares the resolved config against the baseline defaults and
REFUSES TO RUN if the training protocol differs. Overriding a hyperparameter
here without applying the same override to the baselines invalidates the
comparison, so it must be requested explicitly.

Usage
-----
Recommended - proposed model and all baselines in ONE run, one shared config
(guarantees an identical protocol by construction):
    python run_proposed_chililite_v2.py --with-baselines --image-size 224 --target-per-class 410 --epochs 45 --batch-size 32 --measure-latency

Proposed model only, matching the baseline protocol exactly:
    python run_proposed_chililite_v2.py

Change the protocol for EVERY model (still fair, just a different budget):
    python run_baseline_models.py       --epochs 45
    python run_proposed_chililite_v2.py --epochs 45 --allow-unfair-config

Lower-memory setting (note: batch size affects all models equally only if you
apply it to the baselines too):
    python run_proposed_chililite_v2.py --with-baselines --batch-size 16 --num-workers 0

This file must be in the same directory as run_baseline_models.py.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Callable, Dict, Mapping
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from run_baseline_models import (
    BASELINE_MODEL_SPECS,
    ExperimentConfig,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SCRIPT_DIR,
    load_torchvision_model,
    run_cross_validation,
)

PROPOSED_MODEL_NAME = "ChiliLiteV2_MNV3S_MultiLevel_ColorGate"

# Fields that define the training protocol. These must be identical for the
# proposed model and the baselines, otherwise any accuracy difference is
# confounded by the optimisation budget rather than caused by the architecture.
# device/num_workers/measure_latency/output_dir are excluded: they affect
# runtime, not the learned model.
TRAINING_PROTOCOL_FIELDS = (
    "image_size",
    "target_per_class",
    "folds",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "val_fraction",
    "patience",
    "seed",
    "pretrained",
)


class EfficientChannelAttention(nn.Module):
    """Low-cost local cross-channel interaction."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        return x * self.activation(weights)


class LiteMultiScaleRefinement(nn.Module):
    """
    Split channels and apply 3x3/5x5 depthwise convolutions.

    Only depthwise kernels are added, so the parameter increase is very small.
    A learnable residual scale protects the pretrained representation during
    the first fine-tuning epochs.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        first = channels // 2
        second = channels - first
        self.split_sizes = (first, second)

        self.dw3 = nn.Conv2d(
            first,
            first,
            kernel_size=3,
            padding=1,
            groups=first,
            bias=False,
        )
        self.dw5 = nn.Conv2d(
            second,
            second,
            kernel_size=5,
            padding=2,
            groups=second,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.Hardswish(inplace=True)
        self.eca = EfficientChannelAttention(kernel_size=5)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.split(x, self.split_sizes, dim=1)
        refined = torch.cat([self.dw3(first), self.dw5(second)], dim=1)
        refined = self.activation(self.norm(refined))
        refined = self.eca(refined)
        return x + self.residual_scale * refined


class GeMPooling(nn.Module):
    """Learnable generalized-mean pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        return (
            x.clamp_min(self.eps)
            .pow(p)
            .mean(dim=(2, 3))
            .pow(1.0 / p)
        )


class LocalDetailBranch(nn.Module):
    """Compress an intermediate feature map into a small detail vector."""

    def __init__(self, input_channels: int = 48, output_features: int = 64) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, output_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.Hardswish(inplace=True),
            nn.Conv2d(
                output_features,
                output_features,
                kernel_size=3,
                padding=1,
                groups=output_features,
                bias=False,
            ),
            nn.BatchNorm2d(output_features),
            nn.Hardswish(inplace=True),
        )
        self.eca = EfficientChannelAttention(kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(output_features * 2, output_features),
            nn.LayerNorm(output_features),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eca(self.refine(x))
        avg_features = self.avg_pool(x).flatten(1)
        max_features = self.max_pool(x).flatten(1)
        return self.projection(torch.cat([avg_features, max_features], dim=1))


class ColorStatisticsBranch(nn.Module):
    """
    Compute 14 global color descriptors from de-normalized RGB images:

    RGB mean, standard deviation, and skewness: 9
    Brightness mean and standard deviation: 2
    Saturation proxy mean: 1
    Excess-red and excess-green means: 2
    """

    descriptor_count = 14

    def __init__(self, output_features: int = 48) -> None:
        super().__init__()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("normalization_mean", mean, persistent=False)
        self.register_buffer("normalization_std", std, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.descriptor_count, 48),
            nn.LayerNorm(48),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.08),
            nn.Linear(48, output_features),
            nn.LayerNorm(output_features),
            nn.Hardswish(inplace=True),
        )

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        images = (
            normalized_images * self.normalization_std
            + self.normalization_mean
        ).clamp(0.0, 1.0)

        rgb_mean = images.mean(dim=(2, 3))
        rgb_std = images.std(dim=(2, 3), unbiased=False).clamp_min(1e-5)
        centered = images - rgb_mean[:, :, None, None]
        rgb_skew = (
            centered.pow(3).mean(dim=(2, 3))
            / rgb_std.pow(3).clamp_min(1e-5)
        ).clamp(-5.0, 5.0)

        brightness = images.mean(dim=1, keepdim=True)
        brightness_mean = brightness.mean(dim=(2, 3))
        brightness_std = brightness.std(dim=(2, 3), unbiased=False)

        max_channel = images.max(dim=1, keepdim=True).values
        min_channel = images.min(dim=1, keepdim=True).values
        saturation = (max_channel - min_channel) / max_channel.clamp_min(1e-5)
        saturation_mean = saturation.mean(dim=(2, 3))

        red = images[:, 0:1]
        green = images[:, 1:2]
        blue = images[:, 2:3]
        excess_red = (2.0 * red - green - blue).mean(dim=(2, 3))
        excess_green = (2.0 * green - red - blue).mean(dim=(2, 3))

        descriptors = torch.cat(
            [
                rgb_mean,
                rgb_std,
                rgb_skew,
                brightness_mean,
                brightness_std,
                saturation_mean,
                excess_red,
                excess_green,
            ],
            dim=1,
        )

        if descriptors.shape[1] != self.descriptor_count:
            raise RuntimeError(
                f"Expected {self.descriptor_count} descriptors, "
                f"received {descriptors.shape[1]}."
            )
        return self.mlp(descriptors)


class ColorGatedFusion(nn.Module):
    """Use color features to adaptively recalibrate CNN features."""

    def __init__(
        self,
        global_input_features: int,
        local_features: int = 64,
        color_features: int = 48,
        image_output_features: int = 256,
    ) -> None:
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.Linear(global_input_features, 192),
            nn.LayerNorm(192),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.12),
        )

        combined_image_features = 192 + local_features
        if combined_image_features != image_output_features:
            raise ValueError(
                "image_output_features must equal 192 + local_features."
            )

        self.color_gate = nn.Sequential(
            nn.Linear(color_features, image_output_features),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        color_features: torch.Tensor,
    ) -> torch.Tensor:
        global_features = self.global_projection(global_features)
        image_features = torch.cat([global_features, local_features], dim=1)
        gate = self.color_gate(color_features)

        # The gate preserves the original feature and permits a moderate boost.
        gated_image_features = image_features * (1.0 + 0.35 * gate)
        return torch.cat([gated_image_features, color_features], dim=1)


class ChiliLiteV2(nn.Module):
    """Lightweight multi-level, color-guided chili growth-stage classifier."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        base = load_torchvision_model(
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights,
            pretrained,
        )

        feature_blocks = list(base.features.children())

        # Block 8 outputs 48 channels and preserves more local detail.
        self.early_features = nn.Sequential(*feature_blocks[:9])
        self.late_features = nn.Sequential(*feature_blocks[9:])

        final_channels = int(base.classifier[0].in_features)  # 576
        self.local_branch = LocalDetailBranch(
            input_channels=48,
            output_features=64,
        )
        self.final_refinement = LiteMultiScaleRefinement(final_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeMPooling(p=3.0)

        color_features = 48
        self.color_branch = ColorStatisticsBranch(
            output_features=color_features
        )
        self.fusion = ColorGatedFusion(
            global_input_features=final_channels * 2,
            local_features=64,
            color_features=color_features,
            image_output_features=256,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + color_features, 128),
            nn.LayerNorm(128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_map = self.early_features(x)
        final_map = self.late_features(local_map)
        final_map = self.final_refinement(final_map)

        local_features = self.local_branch(local_map)
        avg_features = self.avg_pool(final_map).flatten(1)
        gem_features = self.gem_pool(final_map)
        global_features = torch.cat([avg_features, gem_features], dim=1)

        color_features = self.color_branch(x)
        fused_features = self.fusion(
            global_features,
            local_features,
            color_features,
        )
        return self.classifier(fused_features)


def build_chililite_v2(num_classes: int, pretrained: bool) -> nn.Module:
    return ChiliLiteV2(num_classes=num_classes, pretrained=pretrained)


def build_argument_parser() -> argparse.ArgumentParser:
    # Every default is read from ExperimentConfig, the same dataclass the
    # baselines use. Editing a hyperparameter there changes it for the proposed
    # model and the baselines simultaneously, so they can never diverge.
    defaults = ExperimentConfig()

    parser = argparse.ArgumentParser(
        description=(
            "ChiliLite-V2 MobileNetV3-Small multi-level color-gated model, "
            "trained under the identical protocol used by run_baseline_models.py."
        )
    )
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: results/fair_comparison with --with-baselines, "
            "otherwise results/proposed_lightweight."
        ),
    )
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--target-per-class", type=int, default=defaults.target_per_class)
    parser.add_argument("--folds", type=int, default=defaults.folds)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet" if defaults.pretrained else "none",
    )
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help=(
            "Train the baselines and the proposed model in a single run under one "
            "shared config. This is the recommended mode: identical folds, identical "
            "protocol, and all results land in one summary_metrics.csv."
        ),
    )
    parser.add_argument(
        "--allow-unfair-config",
        action="store_true",
        help=(
            "Permit a training protocol that differs from the baseline defaults. "
            "Only use this when the baselines were re-run with the same overrides."
        ),
    )
    return parser


def audit_training_protocol(cfg: ExperimentConfig, allow_unfair: bool = False) -> None:
    """
    Refuse to run if the proposed model would be trained under a protocol the
    baselines did not receive.

    A longer schedule, a different learning rate, or extra early-stopping
    patience can each produce an accuracy gain on their own. If the proposed
    model gets any of them and the baselines do not, the resulting comparison
    cannot isolate the contribution of the architecture.
    """
    reference = ExperimentConfig()
    deltas = {
        field: (getattr(reference, field), getattr(cfg, field))
        for field in TRAINING_PROTOCOL_FIELDS
        if getattr(reference, field) != getattr(cfg, field)
    }

    if not deltas:
        print(
            "[FAIRNESS] Training protocol is identical to run_baseline_models.py "
            "defaults. Architecture is the only difference."
        )
        return

    lines = "\n".join(
        f"    {field:<18} baseline={baseline!r:<10} -> proposed={proposed!r}"
        for field, (baseline, proposed) in deltas.items()
    )
    message = (
        "The proposed model would use a different training protocol than the "
        f"baselines:\n{lines}"
    )

    if not allow_unfair:
        raise SystemExit(
            "[FAIRNESS ERROR] " + message + "\n\n"
            "  Any accuracy gain would be confounded by the training budget.\n"
            "  Options:\n"
            "    1. Drop the overrides and run: python run_proposed_chililite_v2.py\n"
            "    2. Apply the SAME overrides to the baselines, then re-run this\n"
            "       command with --allow-unfair-config\n"
            "    3. Best: run everything together with --with-baselines"
        )

    warnings.warn(
        "[FAIRNESS] " + message + "\n  Proceeding because --allow-unfair-config was "
        "set. The baselines MUST be re-run with these same settings, otherwise the "
        "comparison is invalid.",
        stacklevel=2,
    )


def select_models(with_baselines: bool) -> Dict[str, Callable[[int, bool], nn.Module]]:
    """Proposed model alone, or every baseline plus the proposed model."""
    if not with_baselines:
        return {PROPOSED_MODEL_NAME: build_chililite_v2}
    # The proposed model is appended last so the baseline column order in
    # summary_metrics.csv stays stable.
    return {**BASELINE_MODEL_SPECS, PROPOSED_MODEL_NAME: build_chililite_v2}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.image_size < 160:
        raise ValueError("--image-size should be at least 160.")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SCRIPT_DIR / (
            "results/fair_comparison"
            if args.with_baselines
            else "results/proposed_lightweight"
        )

    return ExperimentConfig(
        manifest=args.manifest,
        output_dir=output_dir,
        image_size=args.image_size,
        target_per_class=args.target_per_class,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        pretrained=args.weights == "imagenet",
        device=args.device,
        measure_latency=args.measure_latency,
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    cfg = build_config(args)

    # --with-baselines trains everything from one cfg object, so the protocol is
    # shared by construction and the audit is a formality. Standalone runs are
    # audited against the baseline defaults instead.
    if not args.with_baselines:
        audit_training_protocol(cfg, allow_unfair=args.allow_unfair_config)
    else:
        print(
            "[FAIRNESS] Baselines and proposed model share one config object; "
            "identical folds, augmentation, and schedule for every model."
        )

    model_specs = select_models(args.with_baselines)
    print(f"[INFO] Models to train ({len(model_specs)}): {list(model_specs)}")
    print(
        f"[INFO] Protocol: epochs={cfg.epochs} lr={cfg.learning_rate} "
        f"wd={cfg.weight_decay} patience={cfg.patience} "
        f"image_size={cfg.image_size} seed={cfg.seed} folds={cfg.folds}"
    )

    run_cross_validation(model_specs, cfg)


if __name__ == "__main__":
    main()
