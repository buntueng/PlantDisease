"""
Editable proposed lightweight model.

ChiliLiteGFNet is intentionally isolated from the baseline code so that the
architecture can be modified repeatedly without changing the data pipeline,
splits, optimizer, augmentations, metrics, or evaluation code.

Current prototype:
- MobileNetV3-Small feature extractor
- three multi-scale taps
- lightweight 1x1 projections
- ECA-style channel recalibration
- sample-adaptive softmax gates across scales
- depthwise-separable fusion refinement

This is a research prototype, not a claim of novelty by itself. The final paper
should justify each component with ablations and related literature.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ECALayer(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd.")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)                    # B,C,1,1
        y = y.squeeze(-1).transpose(1, 2)   # B,1,C
        y = self.conv(y)
        y = torch.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1) # B,C,1,1
        return x * y


class ProjectionBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, eca_kernel: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
            ECALayer(eca_kernel),
        )


class DepthwiseSeparableRefine(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, eca_kernel: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
            ECALayer(eca_kernel),
        )


class ChiliLiteGFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        tap_indices: tuple[int, ...] = (3, 8, 12),
        tap_channels: tuple[int, ...] = (24, 48, 576),
        projection_channels: int = 96,
        fusion_channels: int = 160,
        gate_hidden_channels: int = 64,
        dropout: float = 0.20,
        eca_kernel_size: int = 3,
    ):
        super().__init__()

        if len(tap_indices) != len(tap_channels):
            raise ValueError("tap_indices and tap_channels must have equal length.")
        if len(tap_indices) < 2:
            raise ValueError("At least two feature taps are required.")

        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )
        backbone = models.mobilenet_v3_small(weights=weights)

        # Keep only convolutional features; discard the original classifier.
        self.features = backbone.features

        self.tap_indices = tuple(int(x) for x in tap_indices)
        self.tap_to_branch = {
            idx: branch for branch, idx in enumerate(self.tap_indices)
        }

        self.projections = nn.ModuleList([
            ProjectionBlock(int(c), projection_channels, eca_kernel_size)
            for c in tap_channels
        ])

        n_branches = len(self.tap_indices)
        self.gate = nn.Sequential(
            nn.Linear(projection_channels * n_branches, gate_hidden_channels),
            nn.SiLU(),
            nn.Linear(gate_hidden_channels, n_branches),
        )

        self.refine = DepthwiseSeparableRefine(
            projection_channels,
            fusion_channels,
            eca_kernel_size,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(fusion_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches: list[torch.Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.tap_to_branch:
                branch_idx = self.tap_to_branch[idx]
                branches.append(self.projections[branch_idx](x))

        if len(branches) != len(self.tap_indices):
            raise RuntimeError(
                f"Expected {len(self.tap_indices)} feature taps, "
                f"but collected {len(branches)}. "
                "Check tap_indices against the installed torchvision version."
            )

        target_size = branches[-1].shape[-2:]
        aligned = [
            F.adaptive_avg_pool2d(b, target_size)
            if b.shape[-2:] != target_size else b
            for b in branches
        ]

        descriptors = torch.cat([
            F.adaptive_avg_pool2d(b, 1).flatten(1)
            for b in aligned
        ], dim=1)

        weights = torch.softmax(self.gate(descriptors), dim=1)

        fused = torch.zeros_like(aligned[0])
        for i, branch in enumerate(aligned):
            gate_i = weights[:, i].view(-1, 1, 1, 1)
            fused = fused + branch * gate_i

        fused = self.refine(fused)
        pooled = self.pool(fused).flatten(1)
        return self.classifier(pooled)


def build_proposed_model(
    cfg: dict[str, Any],
    num_classes: int,
    pretrained: bool | None = None,
) -> nn.Module:
    pcfg = cfg["proposed"]
    if pretrained is None:
        pretrained = bool(cfg["training"]["pretrained"])

    return ChiliLiteGFNet(
        num_classes=num_classes,
        pretrained=pretrained,
        tap_indices=tuple(int(x) for x in pcfg["tap_indices"]),
        tap_channels=tuple(int(x) for x in pcfg["tap_channels"]),
        projection_channels=int(pcfg["projection_channels"]),
        fusion_channels=int(pcfg["fusion_channels"]),
        gate_hidden_channels=int(pcfg["gate_hidden_channels"]),
        dropout=float(pcfg["dropout"]),
        eca_kernel_size=int(pcfg["eca_kernel_size"]),
    )
