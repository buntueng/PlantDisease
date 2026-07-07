"""
Improved lightweight proposed model for chilli leaf disease classification.

This revision keeps the original experiment interface and MobileNetV3-Small
backbone, but changes the fusion head to address weaknesses that can hurt
cross-dataset generalization:

1. MixStyle regularization at early/mid backbone stages during training.
2. GroupNorm-based projection heads (less dependent on batch statistics).
3. Mid-resolution alignment instead of collapsing every tap to the deepest map.
4. Non-competitive residual sigmoid scale gates instead of softmax gates.
5. Gated concatenation rather than a lossy weighted sum.
6. Multi-dilation depthwise refinement for small and large disease patterns.
7. Residual spatial recalibration.
8. Dual global average/max pooling with a stronger lightweight classifier.

The build_proposed_model(...) API is backward-compatible with the original
configuration. New options are read with defaults, so existing config files do
not need to be edited before the first test.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _valid_num_groups(channels: int, preferred: int = 8) -> int:
    """Return the largest group count <= preferred that divides channels."""
    preferred = max(1, min(int(preferred), int(channels)))
    for groups in range(preferred, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ECALayer(nn.Module):
    """Efficient channel recalibration."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be a positive odd integer.")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)                    # B,C,1,1
        y = y.squeeze(-1).transpose(1, 2)   # B,1,C
        y = torch.sigmoid(self.conv(y))
        y = y.transpose(1, 2).unsqueeze(-1) # B,C,1,1
        return x * y


class MixStyle(nn.Module):
    """
    Parameter-free feature-statistics mixing used only during training.

    It perturbs feature style (channel-wise mean/std) while preserving the
    normalized content tensor. This is intended to reduce dependence on
    source-specific appearance statistics.
    """

    def __init__(self, p: float = 0.35, alpha: float = 0.10, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("MixStyle p must be in [0, 1].")
        if alpha <= 0.0:
            raise ValueError("MixStyle alpha must be > 0.")

        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0 or x.shape[0] < 2:
            return x

        if torch.rand((), device=x.device) > self.p:
            return x

        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        batch = x.shape[0]
        perm = torch.randperm(batch, device=x.device)

        # IMPORTANT: sample Beta coefficients in float32. Under CUDA AMP,
        # backbone activations may be float16. PyTorch's Beta sampler is
        # implemented through a Dirichlet sampler, which is not available for
        # Half/BFloat16 in common builds and raises, for example:
        #   NotImplementedError: "dirichlet" not implemented for 'Half'
        # Sampling in FP32 and casting only the sampled coefficients back to
        # the activation dtype keeps MixStyle compatible with AMP.
        concentration = torch.tensor(
            self.alpha,
            dtype=torch.float32,
            device=x.device,
        )
        beta = torch.distributions.Beta(concentration, concentration)
        lam = beta.sample((batch, 1, 1, 1)).to(dtype=x.dtype)

        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]

        return x_norm * sig_mix + mu_mix


class ProjectionBlock(nn.Module):
    """Project each backbone tap to a shared embedding width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eca_kernel: int,
        norm_groups: int = 8,
    ):
        super().__init__()
        groups = _valid_num_groups(out_channels, norm_groups)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.Hardswish(),
            ECALayer(eca_kernel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualScaleGate(nn.Module):
    """
    Sample-adaptive non-competitive gates over scales.

    Unlike a softmax gate, one useful scale does not have to suppress another.
    The final layer is zero-initialized, so every gate starts exactly at 1.0.
    """

    def __init__(
        self,
        channels: int,
        n_branches: int,
        hidden_channels: int,
        dropout: float = 0.10,
    ):
        super().__init__()
        if n_branches < 2:
            raise ValueError("At least two branches are required.")

        descriptor_dim = 2 * channels * n_branches  # avg + max per branch
        self.n_branches = int(n_branches)

        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_channels, n_branches),
        )

        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        descriptors = []
        for branch in branches:
            avg = F.adaptive_avg_pool2d(branch, 1).flatten(1)
            mx = F.adaptive_max_pool2d(branch, 1).flatten(1)
            descriptors.extend([avg, mx])

        descriptor = torch.cat(descriptors, dim=1)
        logits = self.mlp(descriptor)

        # 0.5 + sigmoid(logit) -> (0.5, 1.5), initialized at exactly 1.0.
        return 0.5 + torch.sigmoid(logits)


class MultiDilatedDepthwiseRefine(nn.Module):
    """Lightweight residual refinement at two effective receptive fields."""

    def __init__(
        self,
        channels: int,
        eca_kernel: int,
        norm_groups: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        groups = _valid_num_groups(channels, norm_groups)

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.context = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )

        self.mix = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.Hardswish(),
            ECALayer(eca_kernel),
            nn.Dropout2d(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = self.local(x)
        y_context = self.context(x)
        y = self.mix(torch.cat([y_local, y_context], dim=1))
        return x + y


class ResidualSpatialGate(nn.Module):
    """Spatial recalibration with identity-preserving initialization."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Spatial gate kernel_size must be a positive odd integer.")

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        # sigmoid(0)=0.5 and factor=(0.5+0.5)=1.0 at initialization.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * (0.5 + attn)


class ChiliLiteGFNet(nn.Module):
    """
    Improved ChiliLite-GFNet with cross-scale detail preservation and
    training-time style regularization.

    The class name is intentionally unchanged for compatibility with existing
    experiment runners.
    """

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
        # New options with backward-compatible defaults.
        fusion_target_branch: int = 1,
        mixstyle_p: float = 0.35,
        mixstyle_alpha: float = 0.10,
        mixstyle_indices: tuple[int, ...] = (3, 8),
        branch_dropout: float = 0.10,
        gate_dropout: float = 0.10,
        refine_dropout: float = 0.05,
        norm_groups: int = 8,
        spatial_kernel_size: int = 7,
        classifier_hidden_channels: int | None = None,
    ):
        super().__init__()

        if len(tap_indices) != len(tap_channels):
            raise ValueError("tap_indices and tap_channels must have equal length.")
        if len(tap_indices) < 2:
            raise ValueError("At least two feature taps are required.")
        if len(set(tap_indices)) != len(tap_indices):
            raise ValueError("tap_indices must be unique.")
        if any(b <= a for a, b in zip(tap_indices, tap_indices[1:])):
            raise ValueError("tap_indices must be strictly increasing.")
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0, 1).")

        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        self.tap_indices = tuple(int(x) for x in tap_indices)
        self.tap_to_branch = {
            idx: branch for branch, idx in enumerate(self.tap_indices)
        }

        n_branches = len(self.tap_indices)
        target = int(fusion_target_branch)
        if target < 0:
            target += n_branches
        if not 0 <= target < n_branches:
            raise ValueError(
                f"fusion_target_branch={fusion_target_branch} is invalid for "
                f"{n_branches} branches."
            )
        self.fusion_target_branch = target

        self.projections = nn.ModuleList([
            ProjectionBlock(
                int(c),
                projection_channels,
                eca_kernel_size,
                norm_groups=norm_groups,
            )
            for c in tap_channels
        ])

        # Apply MixStyle only at valid backbone indices.
        self.mixstyle_indices = frozenset(int(i) for i in mixstyle_indices)
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        self.gate = ResidualScaleGate(
            channels=projection_channels,
            n_branches=n_branches,
            hidden_channels=gate_hidden_channels,
            dropout=gate_dropout,
        )
        self.branch_dropout = float(branch_dropout)

        fusion_groups = _valid_num_groups(fusion_channels, norm_groups)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                projection_channels * n_branches,
                fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(fusion_groups, fusion_channels),
            nn.Hardswish(),
        )

        self.refine = MultiDilatedDepthwiseRefine(
            channels=fusion_channels,
            eca_kernel=eca_kernel_size,
            norm_groups=norm_groups,
            dropout=refine_dropout,
        )
        self.spatial_gate = ResidualSpatialGate(spatial_kernel_size)

        hidden = (
            int(classifier_hidden_channels)
            if classifier_hidden_channels is not None
            else max(64, fusion_channels // 2)
        )

        pooled_dim = 2 * fusion_channels  # global average + global max
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(pooled_dim, hidden),
            nn.Hardswish(),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(hidden, num_classes),
        )

    @staticmethod
    def _align_feature(
        x: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Downsample with area pooling; upsample with bilinear interpolation."""
        h, w = x.shape[-2:]
        th, tw = target_size

        if (h, w) == (th, tw):
            return x

        if h >= th and w >= tw:
            return F.adaptive_avg_pool2d(x, target_size)

        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def _apply_branch_dropout(self, gates: torch.Tensor) -> torch.Tensor:
        """Randomly drop complete scale branches during training."""
        if (not self.training) or self.branch_dropout <= 0.0:
            return gates

        keep = (
            torch.rand_like(gates) >= self.branch_dropout
        ).to(gates.dtype)

        # Ensure every sample keeps at least one branch.
        empty = keep.sum(dim=1) == 0
        if empty.any():
            fallback = gates.argmax(dim=1)
            keep[empty, fallback[empty]] = 1.0

        return gates * keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches: list[torch.Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)

            # Training-time appearance-statistics perturbation. Because the
            # tensor is updated here, later backbone layers also learn from it.
            if idx in self.mixstyle_indices:
                x = self.mixstyle(x)

            if idx in self.tap_to_branch:
                branch_idx = self.tap_to_branch[idx]
                branches.append(self.projections[branch_idx](x))

        if len(branches) != len(self.tap_indices):
            raise RuntimeError(
                f"Expected {len(self.tap_indices)} feature taps, "
                f"but collected {len(branches)}. "
                "Check tap_indices/tap_channels against the installed "
                "torchvision MobileNetV3-Small implementation."
            )

        target_size = tuple(
            int(v) for v in branches[self.fusion_target_branch].shape[-2:]
        )
        aligned = [
            self._align_feature(branch, target_size)
            for branch in branches
        ]

        gates = self._apply_branch_dropout(self.gate(aligned))

        gated_branches = [
            branch * gates[:, i].view(-1, 1, 1, 1)
            for i, branch in enumerate(aligned)
        ]

        # Concatenation preserves branch-specific information that a weighted
        # sum would irreversibly collapse.
        fused = self.fuse(torch.cat(gated_branches, dim=1))
        fused = self.refine(fused)
        fused = self.spatial_gate(fused)

        avg = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        mx = F.adaptive_max_pool2d(fused, 1).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)

        return self.classifier(pooled)


def _tuple_of_ints(value: Iterable[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(x) for x in value)


def build_proposed_model(
    cfg: dict[str, Any],
    num_classes: int,
    pretrained: bool | None = None,
) -> nn.Module:
    """
    Backward-compatible model builder.

    Existing config keys are still used. New keys are optional and fall back
    to conservative defaults via dict.get(...).
    """
    pcfg = cfg["proposed"]
    if pretrained is None:
        pretrained = bool(cfg["training"]["pretrained"])

    return ChiliLiteGFNet(
        num_classes=num_classes,
        pretrained=pretrained,
        tap_indices=_tuple_of_ints(pcfg["tap_indices"]),
        tap_channels=_tuple_of_ints(pcfg["tap_channels"]),
        projection_channels=int(pcfg["projection_channels"]),
        fusion_channels=int(pcfg["fusion_channels"]),
        gate_hidden_channels=int(pcfg["gate_hidden_channels"]),
        dropout=float(pcfg["dropout"]),
        eca_kernel_size=int(pcfg["eca_kernel_size"]),
        fusion_target_branch=int(pcfg.get("fusion_target_branch", 1)),
        mixstyle_p=float(pcfg.get("mixstyle_p", 0.35)),
        mixstyle_alpha=float(pcfg.get("mixstyle_alpha", 0.10)),
        mixstyle_indices=_tuple_of_ints(
            pcfg.get("mixstyle_indices", (3, 8))
        ),
        branch_dropout=float(pcfg.get("branch_dropout", 0.10)),
        gate_dropout=float(pcfg.get("gate_dropout", 0.10)),
        refine_dropout=float(pcfg.get("refine_dropout", 0.05)),
        norm_groups=int(pcfg.get("norm_groups", 8)),
        spatial_kernel_size=int(pcfg.get("spatial_kernel_size", 7)),
        classifier_hidden_channels=(
            int(pcfg["classifier_hidden_channels"])
            if pcfg.get("classifier_hidden_channels") is not None
            else None
        ),
    )
