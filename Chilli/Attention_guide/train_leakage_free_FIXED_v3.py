"""
================================================================================
train_leakage_free_fixed.py -- Chili Leaf Disease Classification
Leakage-controlled cross-validation, ablation study, and cost reporting.
================================================================================

HOW TO RUN
    python3 train_leakage_free_fixed.py

Optional examples
    python3 train_leakage_free_fixed.py --dataset "/path/to/original dataset"
    python3 train_leakage_free_fixed.py --epochs 20 --folds 10
    python3 train_leakage_free_fixed.py --augmented-mode auto

DATA SAFETY
    1) ORIGINAL dataset: uses StratifiedKFold and performs augmentation only on
       training batches. This is the preferred and reviewer-safe workflow.

    2) AUGMENTED dataset: the script does NOT randomly split augmented files.
       In auto/grouped mode it first tries to recover the source-image ID from
       filenames, then uses StratifiedGroupKFold so all variants of one source
       stay in the same fold. Training stops if source groups cannot be inferred
       with sufficient confidence.

    The official release reports 1,856 original disease images and 12,000
    augmented disease images. Original images remain the preferred input.

OUTPUTS
    summary_meanstd.csv, per_fold_results.csv, training_history.csv,
    oof_cm_*.csv, class_distribution.csv, dataset_manifest.csv,
    group_inference_report.csv (when augmented grouping is used), checkpoints/

Requires: torch torchvision scikit-learn numpy pandas pillow tqdm
Optional: thop or ptflops (FLOPs), scipy (significance test)
"""

import os
import time
import re
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


# ============================================================================
# CONFIG
# ============================================================================
AUTO_SEARCH_ROOT = "/home/eecommulab/Documents/Po/chili_disease"

CONFIG = {
    # Leave as None to auto-detect under AUTO_SEARCH_ROOT.
    # Or set explicitly, e.g. "/home/eecommulab/.../Original Images"
    "DatasetPath": None,

    "OutputPath":   os.path.join(AUTO_SEARCH_ROOT, "results_leakage_free"),
    "InputSize":    224,
    "BatchSize":    32,
    "LearningRate": 1e-4,
    "Epochs":       20,
    "NumFolds":     10,
    "Seed":         42,
    "NumWorkers":   4,
    "Device":       "cuda" if torch.cuda.is_available() else "cpu",

    # Dataset detection. A folder above this threshold is treated as augmented.
    "MaxImagesGuard": 3000,

    # augmented handling:
    #   "auto"    = prefer original; otherwise infer source groups from filenames
    #   "grouped" = allow filename-inferred grouped CV
    #   "reject"  = require the original dataset and stop on augmented data
    "AugmentedMode": "auto",
    "ExpectedOriginalImages": 1856,
    "MinGroupCompression": 1.25,
    "MaxInferredOriginals": 3500,

    "ModelsToRun": [
        "Proposed_Attn",     # proposed
        "MCCM_only",         # baseline + ablation
        "RNDDNet_only",      # baseline + ablation
        "Concat_noAttn",     # ablation: fusion WITHOUT attention
        "VGG_EffAttnNet",    # baseline
        "MobileNetV2",       # baseline
        "GSAtt_CMNetV3",     # baseline
    ],
}

DISPLAY = {
    "Proposed_Attn": "Proposed Model",
    "Concat_noAttn": "MCCM+RNDDNet (concat)",
    "MCCM_only": "MCCM",
    "RNDDNet_only": "RNDDNet",
    "VGG_EffAttnNet": "VGG-EffAttnNet",
    "MobileNetV2": "MobileNetV2",
    "GSAtt_CMNetV3": "GSAtt-CMNetV3",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
S = CONFIG["InputSize"]
SCRIPT_VERSION = "2026-07-23-empty-folder-fix-v3"


# ============================================================================
# DATASET DISCOVERY + LEAKAGE GUARD
# ============================================================================
# Directories created by this script or by previous runs must never be treated
# as disease classes. torchvision.ImageFolder treats every immediate directory
# as a class, including empty output/checkpoint directories, so we build the
# class index ourselves and keep only directories that actually contain images.
IGNORED_DATA_DIR_NAMES = {
    "results_leakage_free",
    "output_leakage_free",
    "outputs_leakage_free",
    "checkpoints",
    "results",
    "output",
    "outputs",
    "__pycache__",
}


def _is_ignored_data_dir(name):
    name = name.lower().strip()
    return (
        name.startswith(".")
        or name in IGNORED_DATA_DIR_NAMES
        or name.startswith("results_")
        or name.startswith("output_")
        or name.startswith("outputs_")
    )


def _count_images(folder):
    n = 0
    try:
        for e in os.scandir(folder):
            if e.is_file() and e.name.lower().endswith(IMG_EXT):
                n += 1
    except (PermissionError, OSError):
        pass
    return n


def _valid_class_dirs(path):
    """Return immediate class directories containing at least one image."""
    try:
        subs = [
            e for e in os.scandir(path)
            if e.is_dir() and not _is_ignored_data_dir(e.name)
        ]
    except (PermissionError, FileNotFoundError, OSError):
        return []
    return sorted(
        [(e.name, e.path, _count_images(e.path)) for e in subs
         if _count_images(e.path) > 0],
        key=lambda x: x[0].lower(),
    )


def _imagefolder_stats(path):
    classes = _valid_class_dirs(path)
    if len(classes) < 2:
        return None
    per_class = {name: count for name, _, count in classes}
    return len(classes), sum(per_class.values()), per_class


def load_class_folder_index(path):
    """Create ImageFolder-compatible classes/samples while ignoring outputs.

    This avoids torchvision.ImageFolder raising FileNotFoundError when folders
    such as results_leakage_free or output_leakage_free exist inside the
    dataset root but contain no image files.
    """
    class_dirs = _valid_class_dirs(path)
    if len(class_dirs) < 2:
        raise SystemExit(
            f"\nNot enough valid class folders at:\n    {path}\n\n"
            "Expected at least two folders containing image files directly."
        )

    classes = [name for name, _, _ in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples = []
    for class_name, class_path, _ in class_dirs:
        label = class_to_idx[class_name]
        try:
            entries = sorted(os.scandir(class_path), key=lambda e: e.name.lower())
        except (PermissionError, FileNotFoundError, OSError) as exc:
            raise SystemExit(f"Cannot read class folder: {class_path}\n{exc}")
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(IMG_EXT):
                samples.append((entry.path, label))

    if not samples:
        raise SystemExit(f"No supported image files found at: {path}")

    ignored = []
    try:
        for e in os.scandir(path):
            if e.is_dir() and _is_ignored_data_dir(e.name):
                ignored.append(e.name)
            elif e.is_dir() and _count_images(e.path) == 0:
                ignored.append(e.name)
    except OSError:
        pass
    if ignored:
        print("Ignoring non-class directories: " + ", ".join(sorted(set(ignored))))

    return classes, samples


MENDELEY_URL = "https://data.mendeley.com/datasets/w9mr3vf56s/1"


def _dataset_help(extra=""):
    return (
        "\n" + "=" * 78 +
        "\nDATASET SETUP REQUIRED\n" +
        "=" * 78 + "\n" + extra +
        "\nPreferred solution: download the 1,856 ORIGINAL chili leaf disease "
        "images from:\n\n"
        f"    {MENDELEY_URL}\n\n"
        "Expected layout:\n"
        "    <dataset folder>/<class name>/<image files>\n\n"
        "Then run:\n"
        "    python3 train_leakage_free_fixed.py --dataset \"/path/to/original folder\"\n\n"
        "The program will not perform a random image-level split of 12,000 "
        "pre-augmented files because that can place variants of one leaf in "
        "both training and validation folds.\n")


def discover_dataset(root):
    """Return (path, mode, image_count, class_count). Original is preferred."""
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise SystemExit(
            f"\nAUTO_SEARCH_ROOT does not exist:\n    {root}\n\n"
            "Use --dataset /path/to/dataset or edit AUTO_SEARCH_ROOT.")

    found, depth0 = [], root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_ignored_data_dir(d)]
        if dirpath.count(os.sep) - depth0 > 5:
            dirnames[:] = []
            continue
        st = _imagefolder_stats(dirpath)
        if st:
            found.append((dirpath, st[0], st[1]))
            # This is already an ImageFolder root; do not descend into classes.
            dirnames[:] = []

    if not found:
        raise SystemExit(
            f"\nNo class-folder dataset found under:\n    {root}\n\n"
            "Expected: <folder>/<class name>/<images>")

    print(f"Scanning {root}")
    originals, augmented = [], []
    for path, ncls, nimg in sorted(found, key=lambda x: x[2]):
        mode = "original" if nimg <= CONFIG["MaxImagesGuard"] else "augmented"
        print(f"  [{mode.upper():9s}] {nimg:>6d} images, {ncls} classes  {path}")
        (originals if mode == "original" else augmented).append(
            (path, ncls, nimg))

    if originals:
        expected = CONFIG["ExpectedOriginalImages"]
        best = min(originals, key=lambda x: abs(x[2] - expected))
        print(f"\nUsing ORIGINAL: {best[0]}  ({best[2]} images, {best[1]} classes)")
        if abs(best[2] - expected) > 200:
            print(f"WARNING: expected about {expected} originals, but found {best[2]}. "
                  "Verify that the download is complete.")
        return best[0], "original", best[2], best[1]

    if not augmented:
        raise SystemExit(_dataset_help("\nNo usable dataset was found.\n"))

    if CONFIG["AugmentedMode"] == "reject":
        raise SystemExit(_dataset_help(
            "\nOnly an augmented dataset was found and AugmentedMode='reject'.\n"))

    # Pick the candidate closest to the official augmented count.
    best = min(augmented, key=lambda x: abs(x[2] - 12000))
    print(f"\nUsing AUGMENTED candidate: {best[0]} "
          f"({best[2]} images, {best[1]} classes)")
    print("The script will infer source-image groups from filenames before "
          "creating folds. Random image-level splitting is disabled.")
    return best[0], "augmented", best[2], best[1]


def resolve_dataset_path():
    if CONFIG["DatasetPath"]:
        p = os.path.abspath(os.path.expanduser(CONFIG["DatasetPath"]))
        if not os.path.isdir(p):
            raise SystemExit(f"\nDataset path does not exist:\n    {p}")
        st = _imagefolder_stats(p)
        if st is None:
            raise SystemExit(
                f"\nNot a class-folder dataset:\n    {p}\n\n"
                "Expected: <folder>/<class name>/<images>")
        mode = "original" if st[1] <= CONFIG["MaxImagesGuard"] else "augmented"
        if mode == "augmented" and CONFIG["AugmentedMode"] == "reject":
            raise SystemExit(_dataset_help(
                f"\n{st[1]} augmented images found at:\n    {p}\n"))
        print(f"Using {mode.upper()}: {p}  ({st[1]} images, {st[0]} classes)")
        return p, mode, st[1], st[0]
    return discover_dataset(AUTO_SEARCH_ROOT)


# ---- augmented filename grouping ------------------------------------------------
KNOWN_AUGMENTATION_RE = re.compile(
    r"(?i)(?:[_\-\s]+(?:aug(?:mented)?|flip(?:ped)?|hflip|vflip|horizontal|"
    r"vertical|rot(?:ate|ated|ation)?\d*|zoom\d*|shear\d*|shift\d*|"
    r"bright(?:ness)?\d*|contrast\d*|crop\d*|noise\d*|blur\d*))[_\-\s]*.*$")


def _clean_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower().strip()
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def _group_strategy(stem, strategy):
    s = stem
    if strategy == "known_aug_marker":
        s = KNOWN_AUGMENTATION_RE.sub("", s).strip("_")
    elif strategy == "keras_0_number":
        s = re.sub(r"_0_\d+$", "", s)
    elif strategy == "suffix_3_numbers":
        s = re.sub(r"(?:_\d+){3}$", "", s)
    elif strategy == "suffix_2_numbers":
        s = re.sub(r"(?:_\d+){2}$", "", s)
    elif strategy == "suffix_1_number":
        s = re.sub(r"_\d+$", "", s)
    elif strategy == "copy_suffix":
        s = re.sub(r"(?i)(?:_copy)?_?\(?(\d+)\)?$", "", s)
    elif strategy == "before_aug_token":
        s = re.split(r"(?i)(?:^|_)(?:aug|augmented)(?:_|$)", s, maxsplit=1)[0]
    else:
        raise ValueError(strategy)
    return s.strip("_") or stem


def infer_augmented_groups(samples, classes, out_dir):
    """
    Infer original/source IDs from augmented filenames.

    The function evaluates several conservative normalization rules and accepts
    only a strategy that produces a plausible number and size distribution of
    source groups. Labels are included in the group key to prevent accidental
    cross-class merging.
    """
    stems = [_clean_stem(p) for p, _ in samples]
    labels = np.array([y for _, y in samples], dtype=int)
    n = len(samples)
    expected = CONFIG["ExpectedOriginalImages"]
    strategies = [
        "known_aug_marker", "keras_0_number", "suffix_3_numbers",
        "suffix_2_numbers", "suffix_1_number", "copy_suffix",
        "before_aug_token",
    ]

    reports, candidates = [], []
    for strategy in strategies:
        raw_keys = [_group_strategy(stem, strategy) for stem in stems]
        keys = [f"class{label}::{key}" for key, label in zip(raw_keys, labels)]
        counts = pd.Series(keys).value_counts()
        unique_groups = int(counts.size)
        compression = n / max(unique_groups, 1)
        singleton_rate = float((counts == 1).mean())
        max_group = int(counts.max())
        median_group = float(counts.median())
        p95_group = float(counts.quantile(0.95))
        plausible = (
            compression >= CONFIG["MinGroupCompression"] and
            unique_groups <= CONFIG["MaxInferredOriginals"] and
            unique_groups >= len(classes) * CONFIG["NumFolds"] and
            max_group <= 100 and
            p95_group <= 40 and
            median_group >= 1.0
        )
        # Lower is better: prioritize closeness to 1,856 and lower singleton rate.
        score = abs(unique_groups - expected) / max(expected, 1) + singleton_rate
        reports.append({
            "strategy": strategy,
            "unique_groups": unique_groups,
            "compression": compression,
            "singleton_rate": singleton_rate,
            "median_group_size": median_group,
            "p95_group_size": p95_group,
            "max_group_size": max_group,
            "plausible": plausible,
            "score": score,
        })
        if plausible:
            candidates.append((score, strategy, np.asarray(keys, dtype=object)))

    report_df = pd.DataFrame(reports).sort_values(["plausible", "score"],
                                                  ascending=[False, True])
    report_path = os.path.join(out_dir, "group_inference_report.csv")
    report_df.to_csv(report_path, index=False)
    print("\nAugmented filename-group inference:")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nGroup inference report: {report_path}")

    if not candidates:
        raise SystemExit(_dataset_help(
            "\nThe augmented filenames do not preserve a reliable source-image "
            "identifier.\nNo conservative grouping rule passed the safety checks.\n"
            f"See: {report_path}\n"))

    _, strategy, groups = min(candidates, key=lambda x: x[0])
    n_groups = len(np.unique(groups))
    print(f"\n[OK] Selected grouping strategy: {strategy}")
    print(f"[OK] {n:,} augmented files mapped to {n_groups:,} source groups.")
    print("[IMPORTANT] Inspect dataset_manifest.csv and group_inference_report.csv "
          "before reporting the experiment as leakage-free.")
    return groups, strategy


# ============================================================================
# TRANSFORMS  (augmentation on TRAIN only -- this is the leakage fix)
# ============================================================================
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class PathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


# ============================================================================
# ARCHITECTURES
# ============================================================================
class DilatedDenseLayer(nn.Module):
    def __init__(self, in_c, growth, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
            nn.Conv2d(in_c, growth, 3, padding=dilation,
                      dilation=dilation, bias=False))

    def forward(self, x):
        return torch.cat([x, self.conv(x)], 1)


class ResidualDilatedDenseBlock(nn.Module):
    def __init__(self, in_c, num_layers, growth):
        super().__init__()
        self.layers = nn.ModuleList()
        c = in_c
        for i in range(num_layers):
            self.layers.append(DilatedDenseLayer(c, growth, 2 ** (i % 3)))
            c += growth
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, in_c, 1, bias=False))

    def forward(self, x):
        idn = x
        for l in self.layers:
            x = l(x)
        return self.transition(x) + idn


class RNDDNetBackbone(nn.Module):
    """256-d feature vector."""

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.block1 = ResidualDilatedDenseBlock(64, 4, 32)
        self.trans1 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.AvgPool2d(2, 2))
        self.block2 = ResidualDilatedDenseBlock(128, 4, 32)
        self.trans2 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.AvgPool2d(2, 2))
        self.block3 = ResidualDilatedDenseBlock(256, 4, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 256

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        return torch.flatten(self.pool(x), 1)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        b = out_c // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_c, b, 1, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_c, b, 1, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True),
                                nn.Conv2d(b, b, 3, padding=1, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, b, 1, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True),
                                nn.Conv2d(b, b, 5, padding=2, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                nn.Conv2d(in_c, b, 1, bias=False),
                                nn.BatchNorm2d(b), nn.ReLU(True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class MCCMBackbone(nn.Module):
    """512-d feature vector."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = MultiScaleBlock(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = MultiScaleBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = MultiScaleBlock(256, 512)
        self.gpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.layer3(x)
        return torch.flatten(self.gpool(x), 1)


class FeatureAttention(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(in_f, in_f // 4), nn.ReLU(True),
                                 nn.Linear(in_f // 4, in_f), nn.Sigmoid())

    def forward(self, x):
        return x * self.att(x)


def make_head(in_f, nc):
    return nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(True),
                         nn.Dropout(0.2), nn.Linear(256, nc))


class SingleBranch(nn.Module):
    def __init__(self, backbone, nc):
        super().__init__()
        self.backbone = backbone
        self.head = make_head(backbone.out_dim, nc)

    def forward(self, x):
        return self.head(self.backbone(x))


class FusionModel(nn.Module):
    def __init__(self, nc, use_attention):
        super().__init__()
        self.mccm = MCCMBackbone()
        self.rndd = RNDDNetBackbone()
        d = self.mccm.out_dim + self.rndd.out_dim          # 768
        self.attention = FeatureAttention(d) if use_attention else nn.Identity()
        self.head = make_head(d, nc)

    def forward(self, x):
        f = torch.cat([self.mccm(x), self.rndd(x)], 1)
        return self.head(self.attention(f))


class EfficientAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.kc, self.vc = in_channels // 8, in_channels
        self.keys = nn.Conv2d(in_channels, self.kc, 1)
        self.queries = nn.Conv2d(in_channels, self.kc, 1)
        self.values = nn.Conv2d(in_channels, self.vc, 1)
        self.reproj = nn.Conv2d(self.vc, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        k = F.softmax(self.keys(x).view(b, self.kc, h * w), dim=2)
        q = F.softmax(self.queries(x).view(b, self.kc, h * w), dim=1)
        v = self.values(x).view(b, self.vc, h * w)
        ctx = torch.bmm(k, v.transpose(1, 2))
        att = torch.bmm(ctx.transpose(1, 2), q).view(b, self.vc, h, w)
        return x + self.reproj(att)


class VGG_EffAttnNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.attention = EfficientAttention(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True),
                                        nn.Dropout(0.4), nn.Linear(256, nc))

    def forward(self, x):
        x = self.attention(self.features(x))
        return self.classifier(torch.flatten(self.pool(x), 1))


class GlobalSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, C, w, h = x.size()
        q = self.query(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key(x).view(b, -1, w * h)
        att = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, -1, w * h)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(b, C, w, h)
        return self.gamma * out + x


class GSAtt_CMNetV3(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = models.mobilenet_v3_large(weights=None).features
        self.attention = GlobalSelfAttention(960)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(960, 1280), nn.Hardswish(True),
                                        nn.Dropout(0.2), nn.Linear(1280, nc))

    def forward(self, x):
        x = self.attention(self.features(x))
        return self.classifier(torch.flatten(self.pool(x), 1))


def MODEL_FACTORY(name, nc):
    if name == "Proposed_Attn":
        return FusionModel(nc, True)
    if name == "Concat_noAttn":
        return FusionModel(nc, False)
    if name == "MCCM_only":
        return SingleBranch(MCCMBackbone(), nc)
    if name == "RNDDNet_only":
        return SingleBranch(RNDDNetBackbone(), nc)
    if name == "VGG_EffAttnNet":
        return VGG_EffAttnNet(nc)
    if name == "MobileNetV2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, nc)
        return m
    if name == "GSAtt_CMNetV3":
        return GSAtt_CMNetV3(nc)
    raise ValueError(f"Unknown model '{name}'")


# ============================================================================
# METRICS
# ============================================================================
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_flops(m, device):
    x = torch.randn(1, 3, S, S).to(device)
    try:
        from thop import profile
        f, _ = profile(m, inputs=(x,), verbose=False)
        return f
    except Exception:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                m, (3, S, S), as_strings=False,
                print_per_layer_stat=False, verbose=False)
            return macs * 2
        except Exception:
            return float("nan")


@torch.no_grad()
def inference_ms(m, device, n=200):
    m.eval()
    x = torch.randn(1, 3, S, S).to(device)
    for _ in range(20):
        m(x)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        m(x)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def multiclass_tnr(cm):
    FP = cm.sum(0) - np.diag(cm)
    FN = cm.sum(1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return np.divide(TN, TN + FP, out=np.zeros_like(TN, float),
                     where=(TN + FP) != 0).mean()


# ============================================================================
# FOLDS
# ============================================================================
def _validate_folds(folds, n_samples, groups=None):
    seen = set()
    for k, (tr, va) in enumerate(folds):
        tr_set, va_set = set(map(int, tr)), set(map(int, va))
        assert not (tr_set & va_set), f"fold {k+1}: train/val sample overlap"
        assert not (seen & va_set), f"fold {k+1}: overlaps a previous fold"
        seen |= va_set
        if groups is not None:
            tr_groups = set(groups[tr])
            va_groups = set(groups[va])
            overlap = tr_groups & va_groups
            assert not overlap, (
                f"fold {k+1}: {len(overlap)} source groups occur in both train/val")
    assert len(seen) == n_samples, "folds do not validate every image exactly once"


def build_folds(samples, labels, groups=None, dataset_mode="original"):
    requested = int(CONFIG["NumFolds"])
    class_counts = np.bincount(labels)
    max_by_class = int(class_counts.min())
    n_splits = min(requested, max_by_class)

    if dataset_mode == "augmented":
        if StratifiedGroupKFold is None:
            raise SystemExit(
                "\nStratifiedGroupKFold is unavailable. Upgrade scikit-learn:\n"
                "    pip install -U scikit-learn\n")
        # Each class must contain enough distinct groups.
        per_class_groups = []
        for c in np.unique(labels):
            per_class_groups.append(len(np.unique(groups[labels == c])))
        n_splits = min(n_splits, min(per_class_groups))

    if n_splits < 2:
        raise SystemExit("\nNot enough samples/groups per class for cross-validation.")
    if n_splits != requested:
        print(f"WARNING: NumFolds reduced from {requested} to {n_splits} because "
              "the smallest class does not support the requested value.")
        CONFIG["NumFolds"] = n_splits

    if dataset_mode == "augmented":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=CONFIG["Seed"])
        folds = list(splitter.split(np.arange(len(samples)), labels, groups))
        _validate_folds(folds, len(samples), groups)
        print(f"\n[OK] {n_splits} StratifiedGroupKFold folds.")
        print("[OK] No source group occurs in both training and validation.")
        print("[OK] Identical grouped folds are reused for every model.\n")
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=CONFIG["Seed"])
        folds = list(splitter.split(np.arange(len(samples)), labels))
        _validate_folds(folds, len(samples))
        print(f"\n[OK] {n_splits} StratifiedKFold folds.")
        print("[OK] Every original image is validated exactly once.")
        print("[OK] Augmentation is applied to training batches only.")
        print("[OK] Identical folds are reused for every model.\n")
    return folds


# ============================================================================
# TRAIN / EVAL
# ============================================================================
def run_model(name, samples, folds, num_classes, out_dir):
    device = CONFIG["Device"]
    per_fold, history, oof_t, oof_p = [], [], [], []

    probe = MODEL_FACTORY(name, num_classes).to(device)
    n_par = count_params(probe)
    flops = count_flops(probe, device)
    inf_ms = inference_ms(probe, device)
    del probe
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_ld = DataLoader(
            PathDataset([samples[i] for i in tr_idx], train_transform),
            batch_size=CONFIG["BatchSize"], shuffle=True,
            num_workers=CONFIG["NumWorkers"], pin_memory=True)
        va_ld = DataLoader(
            PathDataset([samples[i] for i in va_idx], eval_transform),
            batch_size=CONFIG["BatchSize"], shuffle=False,
            num_workers=CONFIG["NumWorkers"], pin_memory=True)

        model = MODEL_FACTORY(name, num_classes).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=CONFIG["LearningRate"])

        t0 = time.time()
        for ep in range(CONFIG["Epochs"]):
            model.train()
            rl = c = n = 0
            for x, y in tqdm(tr_ld, desc=f"{DISPLAY[name]} f{k+1} e{ep+1}",
                             leave=False, unit="b"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                rl += loss.item() * x.size(0)
                n += y.size(0)
                c += (out.argmax(1) == y).sum().item()
            tr_loss, tr_acc = rl / n, c / n

            model.eval()
            vl = vc = vn = 0
            with torch.no_grad():
                for x, y in va_ld:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    vl += crit(out, y).item() * x.size(0)
                    vn += y.size(0)
                    vc += (out.argmax(1) == y).sum().item()
            history.append({"Model": name, "Fold": k + 1, "Epoch": ep + 1,
                            "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": vl / vn, "val_acc": vc / vn})
        ttime = time.time() - t0

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y in va_ld:
                yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
                yt.extend(y.numpy())
        yt, yp = np.array(yt), np.array(yp)
        oof_t.extend(yt)
        oof_p.extend(yp)
        cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))

        per_fold.append({
            "Model": name, "Fold": k + 1,
            "Accuracy": accuracy_score(yt, yp),
            "Recall": recall_score(yt, yp, average="macro", zero_division=0),
            "TNR": multiclass_tnr(cm),
            "F1": f1_score(yt, yp, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(yt, yp),
            "TrainTime_s": ttime})
        print(f"  {DISPLAY[name]:<24s} fold {k+1:2d}/{CONFIG['NumFolds']}  "
              f"acc={per_fold[-1]['Accuracy']:.4f}  f1={per_fold[-1]['F1']:.4f}")

        if k == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "checkpoints", f"{name}_fold1.pt"))
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(per_fold)
    oof_cm = confusion_matrix(oof_t, oof_p, labels=list(range(num_classes)))
    np.savetxt(os.path.join(out_dir, f"oof_cm_{name}.csv"), oof_cm,
               fmt="%d", delimiter=",")

    summary = {"Model": name, "Display": DISPLAY[name],
               "Params_M": n_par / 1e6, "FLOPs_G": flops / 1e9,
               "Infer_ms": inf_ms,
               "OOF_Accuracy": accuracy_score(oof_t, oof_p),
               "TrainTime_mean_s": df["TrainTime_s"].mean()}
    for m in ["Accuracy", "Recall", "TNR", "F1", "MCC"]:
        summary[f"{m}_mean"] = df[m].mean()
        summary[f"{m}_std"] = df[m].std()
    return df, pd.DataFrame(history), summary


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Leakage-controlled chili leaf disease experiments")
    p.add_argument("--dataset", default=None,
                   help="ImageFolder root: <dataset>/<class>/<images>")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--augmented-mode", choices=["auto", "grouped", "reject"],
                   default=None,
                   help="How to handle a pre-augmented dataset")
    p.add_argument("--models", default=None,
                   help="Comma-separated model names from CONFIG['ModelsToRun']")
    return p.parse_args()


def apply_args(args):
    if args.dataset:
        CONFIG["DatasetPath"] = args.dataset
    if args.output:
        CONFIG["OutputPath"] = args.output
    if args.epochs is not None:
        CONFIG["Epochs"] = args.epochs
    if args.folds is not None:
        CONFIG["NumFolds"] = args.folds
    if args.batch_size is not None:
        CONFIG["BatchSize"] = args.batch_size
    if args.workers is not None:
        CONFIG["NumWorkers"] = args.workers
    if args.augmented_mode:
        CONFIG["AugmentedMode"] = args.augmented_mode
    if args.models:
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        unknown = [x for x in requested if x not in DISPLAY]
        if unknown:
            raise SystemExit(f"Unknown model name(s): {', '.join(unknown)}")
        CONFIG["ModelsToRun"] = requested


def main():
    apply_args(parse_args())

    print("=" * 74)
    print("  Chili Leaf Disease -- leakage-controlled experiments")
    print(f"  Script version: {SCRIPT_VERSION}")
    print("=" * 74)
    print(f"Device: {CONFIG['Device']}")
    if str(CONFIG["Device"]).startswith("cuda"):
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Mode  : augmented={CONFIG['AugmentedMode']}, "
          f"folds={CONFIG['NumFolds']}, epochs={CONFIG['Epochs']}")
    print()

    dataset_path, dataset_mode, _, _ = resolve_dataset_path()
    out_dir = os.path.abspath(CONFIG["OutputPath"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    torch.manual_seed(CONFIG["Seed"])
    np.random.seed(CONFIG["Seed"])
    if str(CONFIG["Device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(CONFIG["Seed"])
        torch.backends.cudnn.benchmark = True

    classes, samples = load_class_folder_index(dataset_path)
    num_classes = len(classes)
    labels = np.array([label for _, label in samples], dtype=int)

    print(f"\nClasses ({num_classes}):")
    dist = []
    for i, class_name in enumerate(classes):
        n = int((labels == i).sum())
        dist.append({"Class": class_name, "Images": n})
        print(f"   {class_name:<28s} {n:>5d}")
    pd.DataFrame(dist).to_csv(
        os.path.join(out_dir, "class_distribution.csv"), index=False)

    grouping_strategy = "unique_original_file"
    if dataset_mode == "augmented":
        groups, grouping_strategy = infer_augmented_groups(samples, classes, out_dir)
    else:
        groups = np.asarray([f"original::{i}" for i in range(len(samples))],
                            dtype=object)

    manifest = pd.DataFrame({
        "path": [p for p, _ in samples],
        "class": [classes[y] for _, y in samples],
        "label": labels,
        "source_group": groups,
        "dataset_mode": dataset_mode,
        "grouping_strategy": grouping_strategy,
    })
    manifest.to_csv(os.path.join(out_dir, "dataset_manifest.csv"), index=False)

    folds = build_folds(samples, labels, groups=groups,
                        dataset_mode=dataset_mode)

    all_f, all_h, all_s = [], [], []
    for name in CONFIG["ModelsToRun"]:
        print(f"\n{'='*62}\n  {DISPLAY[name]}\n{'='*62}")
        df, hist, summ = run_model(name, samples, folds, num_classes, out_dir)
        all_f.append(df)
        all_h.append(hist)
        all_s.append(summ)
        # Save after each model so completed work survives an interruption.
        pd.concat(all_f).to_csv(
            os.path.join(out_dir, "per_fold_results.csv"), index=False)
        pd.concat(all_h).to_csv(
            os.path.join(out_dir, "training_history.csv"), index=False)
        pd.DataFrame(all_s).to_csv(
            os.path.join(out_dir, "summary_meanstd.csv"), index=False)
        print(f"  -> {DISPLAY[name]}: {summ['Accuracy_mean']:.4f} "
              f"+/- {summ['Accuracy_std']:.4f}  "
              f"(OOF {summ['OOF_Accuracy']:.4f})")

    print("\n" + "=" * 74)
    print("  SUMMARY (mean +/- std across folds)")
    print("=" * 74)
    for s in sorted(all_s, key=lambda r: -r["Accuracy_mean"]):
        print(f"{s['Display']:>24s} | acc {s['Accuracy_mean']:.4f}"
              f"+/-{s['Accuracy_std']:.4f} | F1 {s['F1_mean']:.4f} "
              f"| MCC {s['MCC_mean']:.4f} | OOF {s['OOF_Accuracy']:.4f} "
              f"| {s['Params_M']:.2f}M | {s['Infer_ms']:.2f} ms")

    sd = {s["Model"]: s for s in all_s}
    if "Concat_noAttn" in sd and "Proposed_Attn" in sd:
        a, b = sd["Concat_noAttn"], sd["Proposed_Attn"]
        gap = b["Accuracy_mean"] - a["Accuracy_mean"]
        print("\n" + "-" * 74)
        print("  DOES THE ATTENTION MODULE HELP?  (key ablation)")
        print("-" * 74)
        print(f"  concat without attention : {a['Accuracy_mean']:.4f} "
              f"+/- {a['Accuracy_std']:.4f}")
        print(f"  with attention (proposed): {b['Accuracy_mean']:.4f} "
              f"+/- {b['Accuracy_std']:.4f}")
        print(f"  difference               : {gap:+.4f}")
        try:
            from scipy import stats
            fdf = pd.concat(all_f)
            x = fdf[fdf.Model == "Concat_noAttn"].sort_values("Fold")["Accuracy"].values
            y = fdf[fdf.Model == "Proposed_Attn"].sort_values("Fold")["Accuracy"].values
            t, p = stats.ttest_rel(y, x)
            print(f"  paired t-test            : t={t:.3f}, p={p:.4f}")
            if p < 0.05 and gap > 0:
                print("  => significant. Report the p-value in the paper.")
            else:
                print("  => NOT significant. Do not claim an accuracy improvement;")
                print("     report the null result in the Discussion.")
        except ImportError:
            print("  (pip install scipy for the significance test)")

    print(f"\nAll results written to: {out_dir}")
    print("Files: summary_meanstd.csv, per_fold_results.csv,")
    print("       training_history.csv, oof_cm_*.csv, class_distribution.csv,")
    print("       dataset_manifest.csv, group_inference_report.csv (if used)")


if __name__ == "__main__":
    main()
