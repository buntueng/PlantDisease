#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 train_all_models.py
 Consolidated, leakage-free training + evaluation for the soybean leaf disease
 study (EfficientNetV2-S + Selective Kernel attention).

 Runs every model from a single entry point and records *everything* the
 manuscript revision needs.

 WHAT THIS SCRIPT PRODUCES  ->  WHERE IT GOES IN THE PAPER
 --------------------------------------------------------------------------
 per_fold_results.csv (per model)      -> mean +/- std for Table (backbones)
 all_models_summary.csv                -> backbone table + attention ablation
 training_history.csv (per model)      -> training-curve figure
 oof_predictions.npz (per model)       -> aggregated confusion matrix
 edge_profile.json (per model)         -> edge-deployment table (params/size/FPS)
 cross_dataset/*.npz + metrics.json    -> cross-dataset external-validation tables
 (significance test is computed in the companion notebook from per_fold CSVs)

 IMPORTANT FIXES vs. the original scripts
 --------------------------------------------------------------------------
 1) NO DATA LEAKAGE. Point ORIGINAL_DATASET_PATH at the *unbalanced* ASDID.
    The 10-fold split is done on the original images; the controlled
    up-sampling is applied ONLY inside each training fold. Validation folds
    contain original images only.
 2) Metrics use the BEST-epoch predictions (the checkpoint actually saved),
    not the last epoch.
 3) Adds ONNX export + edge profiling, out-of-fold prediction saving, and a
    zero-shot cross-dataset evaluation (open + constrained protocols).

 NOTE ON THE "DUAL" BASELINE
 --------------------------------------------------------------------------
 The DualAttention module below is the CBAM design from your original dual.py
 (channel MLP over avg+max pool, then a 7x7 spatial conv). This is NOT the
 DANet "Dual Attention" whose equations appear in the manuscript. Either
 (a) relabel this baseline as CBAM and cite Woo et al. (2018), or
 (b) replace this module with a true DANet position+channel module.
 Decide before submission so the text and code agree.
================================================================================
"""

import os
import csv
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import label_binarize

# =============================================================================
# CONFIGURATION  -- edit these paths/flags for your machine
# =============================================================================
# Point this at the ORIGINAL (unbalanced) ASDID, NOT the pre-balanced folder.
ORIGINAL_DATASET_PATH = '/home/bt/Desktop/Bee/soybean/ASDID_original_224'

# External dataset for cross-dataset validation (Mendeley 6fhphxg297).
# Set to None to skip the cross-dataset stage.
EXTERNAL_DATASET_PATH = '/home/bt/Desktop/Bee/soybean/mendeley_soybean'

RESULTS_ROOT = '/home/bt/Desktop/Bee/soybean/results_consolidated'

BATCH_SIZE     = 32
NUM_EPOCHS     = 20
NUM_CLASSES    = 8
NUM_FOLDS      = 10
LEARNING_RATE  = 1e-4
NUM_WORKERS    = 4
SEED           = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Which models to run. Comment out any you do not need.
MODELS_TO_RUN = [
    # --- backbone comparison ---
    'efficientnet_v2_s',
    'mnasnet1_0',
    'mobilenet_v3_small',
    'shufflenet_v2',
    'squeezenet',
    # --- attention ablation on EfficientNetV2-S ---
    'effv2s_sknet',     # <-- PROPOSED model
    'effv2s_dual',      # CBAM (see note above)
    'effv2s_shuffle',
    'effv2s_simam',
]
PROPOSED_MODEL = 'effv2s_sknet'   # used for cross-dataset eval + OOF confusion

# Quick smoke test: 1 fold, 1 epoch (set to False for the real run).
QUICK_TEST = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# ATTENTION MODULES (inserted after EfficientNetV2-S .features, 1280 channels)
# =============================================================================
class SKAttention(nn.Module):
    """Selective Kernel attention (proposed). Two dilated branches fused by
    softmax-weighted selection."""
    def __init__(self, channel=1280, branches=2, reduction=16, L=32):
        super().__init__()
        d = max(int(channel / reduction), L)
        self.branches = branches
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1 + i,
                          dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
            ) for i in range(branches)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList(
            [nn.Conv2d(d, channel, kernel_size=1, bias=False) for _ in range(branches)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        U = sum(conv_outs)
        Z = self.fc(self.gap(U))
        weights = torch.stack([fc(Z) for fc in self.fcs], dim=1)
        attention_weights = self.softmax(weights)
        return sum(attention_weights[:, i] * conv_outs[i] for i in range(self.branches))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class DualAttention(nn.Module):
    """NOTE: this is CBAM (channel + spatial), as in your original dual.py.
    See the header note on aligning this with the manuscript."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ShuffleAttention(nn.Module):
    def __init__(self, channel=1280, G=8):
        super().__init__()
        self.G = G
        self.sub_channel = channel // (2 * G)
        self.weight_c = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.bias_c = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.weight_s = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.bias_s = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(self.sub_channel, self.sub_channel)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, -1, h, w)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        x_c = self.avg_pool(x_0)
        x_0 = x_0 * self.sigmoid(self.weight_c * x_c + self.bias_c)
        x_s = self.gn(x_1)
        x_1 = x_1 * self.sigmoid(self.weight_s * x_s + self.bias_s)
        out = torch.cat([x_0, x_1], dim=1).view(b, -1, h, w)
        return self.channel_shuffle(out, 2)


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(v)


class EffV2S_WithAttention(nn.Module):
    """EfficientNetV2-S backbone + an attention module before pooling."""
    def __init__(self, attention: nn.Module, num_classes=NUM_CLASSES):
        super().__init__()
        base = models.efficientnet_v2_s(weights='DEFAULT')
        self.features = base.features
        self.attention = attention
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =============================================================================
# MODEL REGISTRY:  name -> (builder, group)
# group is 'backbone' or 'attention' (for organising the two result tables)
# =============================================================================
def _efficientnet_v2_s():
    m = models.efficientnet_v2_s(weights='DEFAULT')
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

def _mnasnet1_0():
    m = models.mnasnet1_0(weights='DEFAULT')
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

def _mobilenet_v3_small():
    m = models.mobilenet_v3_small(weights='DEFAULT')
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m

def _shufflenet_v2():
    m = models.shufflenet_v2_x1_0(weights='DEFAULT')
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def _squeezenet():
    m = models.squeezenet1_0(weights='DEFAULT')
    m.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
    m.num_classes = NUM_CLASSES
    return m

MODEL_REGISTRY = {
    'efficientnet_v2_s':  (_efficientnet_v2_s, 'backbone'),
    'mnasnet1_0':         (_mnasnet1_0, 'backbone'),
    'mobilenet_v3_small': (_mobilenet_v3_small, 'backbone'),
    'shufflenet_v2':      (_shufflenet_v2, 'backbone'),
    'squeezenet':         (_squeezenet, 'backbone'),
    'effv2s_sknet':       (lambda: EffV2S_WithAttention(SKAttention(1280, branches=2)), 'attention'),
    'effv2s_dual':        (lambda: EffV2S_WithAttention(DualAttention(1280)), 'attention'),
    'effv2s_shuffle':     (lambda: EffV2S_WithAttention(ShuffleAttention(1280, G=8)), 'attention'),
    'effv2s_simam':       (lambda: EffV2S_WithAttention(SimAM()), 'attention'),
}


# =============================================================================
# TRANSFORMS
# =============================================================================
def base_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def aug_transform(level):
    """level 1 = rotation; level 2 = rotation + contrast (per the manuscript)."""
    ops = [transforms.Resize((224, 224)), transforms.RandomRotation(25)]
    if level >= 2:
        ops.append(transforms.ColorJitter(contrast=0.3))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


# =============================================================================
# LEAKAGE-FREE CONTROLLED UP-SAMPLING (pure-python plan; unit-tested)
# =============================================================================
def build_oversampled_plan(train_indices, targets):
    """Return (plan, c_factors).
    plan: list of (original_index, aug_level) entries for the training fold.
          Each original sample appears once at level 0 (no aug); minority
          classes get (C_i - 1) extra copies augmented at the class level.
    c_factors: {class_label: C_i} with C_i = floor(N_max / N_i).
    """
    cls_to_indices = defaultdict(list)
    for idx in train_indices:
        cls_to_indices[int(targets[idx])].append(int(idx))
    counts = {c: len(v) for c, v in cls_to_indices.items()}
    n_max = max(counts.values())

    c_factors, plan = {}, []
    for c, idxs in cls_to_indices.items():
        c_i = max(1, n_max // counts[c])
        c_factors[c] = c_i
        aug_level = 0 if c_i == 1 else (1 if c_i == 2 else 2)
        for idx in idxs:
            plan.append((idx, 0))                  # original, no augmentation
            for _ in range(c_i - 1):
                plan.append((idx, aug_level))      # augmented duplicate(s)
    return plan, c_factors


class PlanDataset(Dataset):
    """Materialises an up-sampling plan with per-entry augmentation level.
    `pil_dataset` must be an ImageFolder created with transform=None."""
    def __init__(self, pil_dataset, plan):
        self.ds = pil_dataset
        self.plan = plan
        self.base_tf = base_transform()
        self.aug_tfs = {1: aug_transform(1), 2: aug_transform(2)}

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        idx, level = self.plan[i]
        img, label = self.ds[idx]            # PIL image, int label
        tf = self.base_tf if level == 0 else self.aug_tfs[level]
        return tf(img), label


class EvalSubset(Dataset):
    """Validation subset: original images only, base transform, no oversampling."""
    def __init__(self, pil_dataset, indices):
        self.ds = pil_dataset
        self.indices = list(indices)
        self.base_tf = base_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.ds[self.indices[i]]
        return self.base_tf(img), label


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - (fp + fn + np.diag(cm))
    spec = tn.sum() / (tn.sum() + fp.sum()) if (tn.sum() + fp.sum()) > 0 else 0.0
    out = {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'F1':          f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision':   precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Specificity': spec,
        'MCC':         matthews_corrcoef(y_true, y_pred),
    }
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        out['AUC'] = roc_auc_score(y_bin, y_prob, multi_class='ovr')
    except Exception:
        out['AUC'] = float('nan')
    return out


# =============================================================================
# EDGE PROFILING (ONNX export + params + latency/FPS)
# =============================================================================
@torch.no_grad()
def profile_edge(model, model_dir, name):
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    size_mb_params = n_params * 4 / 1e6   # float32

    # ONNX export (artifact + on-disk size). Non-fatal if it fails.
    onnx_path = os.path.join(model_dir, f'{name}.onnx')
    onnx_size_mb = None
    try:
        dummy = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(model, dummy, onnx_path, opset_version=13,
                          input_names=['input'], output_names=['logits'],
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}})
        onnx_size_mb = os.path.getsize(onnx_path) / 1e6
    except Exception as e:
        print(f"  [edge] ONNX export failed for {name}: {e}")

    def measure_latency(dev, n_warmup=10, n_runs=50):
        m = model.to(dev)
        x = torch.randn(1, 3, 224, 224, device=dev)
        for _ in range(n_warmup):
            m(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            m(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.time() - t0) / n_runs * 1000.0
        return ms, 1000.0 / ms

    cpu_ms, cpu_fps = measure_latency(torch.device('cpu'))
    prof = {
        'parameters_millions': round(n_params / 1e6, 3),
        'param_size_mb': round(size_mb_params, 3),
        'onnx_size_mb': round(onnx_size_mb, 3) if onnx_size_mb else None,
        'cpu_latency_ms': round(cpu_ms, 3),
        'cpu_fps': round(cpu_fps, 2),
    }
    if torch.cuda.is_available():
        gpu_ms, gpu_fps = measure_latency(torch.device('cuda'))
        prof['gpu_latency_ms'] = round(gpu_ms, 3)
        prof['gpu_fps'] = round(gpu_fps, 2)
    model.to(device)
    with open(os.path.join(model_dir, 'edge_profile.json'), 'w') as f:
        json.dump(prof, f, indent=2)
    return prof


# =============================================================================
# TRAIN + EVALUATE ONE MODEL ACROSS FOLDS
# =============================================================================
def run_model(name, pil_dataset, targets, splits):
    builder, group = MODEL_REGISTRY[name]
    model_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n{'='*70}\nMODEL: {name}  ({group})\n{'='*70}")

    per_fold_rows, history_rows = [], []
    oof_true, oof_pred, oof_prob = [], [], []
    overall_best_acc, best_state = 0.0, None

    n_epochs = 1 if QUICK_TEST else NUM_EPOCHS

    for fold, (train_ids, val_ids) in enumerate(splits):
        plan, c_factors = build_oversampled_plan(train_ids, targets)
        if fold == 0:
            print(f"  up-sampling C_i per class: {c_factors}")
        train_ds = PlanDataset(pil_dataset, plan)
        val_ds = EvalSubset(pil_dataset, val_ids)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        model = builder().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_fold_acc = 0.0
        best_fold_pack = None
        fold_start = time.time()

        for epoch in range(n_epochs):
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in tqdm(train_loader,
                                       desc=f"{name} F{fold+1} E{epoch+1} [train]",
                                       leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            # validation
            model.eval()
            val_loss, val_correct = 0.0, 0
            yl, yp, ypr = [], [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
                    probs = torch.softmax(outputs, 1)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    yl.extend(labels.cpu().numpy())
                    yp.extend(preds.cpu().numpy())
                    ypr.extend(probs.cpu().numpy())

            tr_acc = train_correct / len(train_ds)
            va_acc = val_correct / len(val_ds)
            history_rows.append([fold + 1, epoch + 1,
                                 train_loss / len(train_ds), tr_acc,
                                 val_loss / len(val_ds), va_acc])
            print(f"  F{fold+1} E{epoch+1}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

            # capture BEST-epoch predictions (matches the saved checkpoint)
            if va_acc > best_fold_acc:
                best_fold_acc = va_acc
                best_fold_pack = (np.array(yl), np.array(yp), np.array(ypr))
                torch.save(model.state_dict(),
                           os.path.join(model_dir, f'{name}_fold{fold+1}_best.pth'))
                if va_acc > overall_best_acc:
                    overall_best_acc = va_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ---- best-epoch metrics for this fold ----
        fold_time = time.time() - fold_start
        yt, ypd, ypb = best_fold_pack
        m = compute_metrics(yt, ypd, ypb, NUM_CLASSES)
        per_fold_rows.append([fold + 1, m['Accuracy'], m['F1'], m['Sensitivity'],
                              m['Precision'], m['Specificity'], m['MCC'], m['AUC'],
                              fold_time])
        oof_true.append(yt); oof_pred.append(ypd); oof_prob.append(ypb)

    # ---- save per-fold + history + OOF + best checkpoint ----
    cols = ['Fold', 'Accuracy', 'F1', 'Sensitivity', 'Precision',
            'Specificity', 'MCC', 'AUC', 'TrainTime_s']
    pf = pd.DataFrame(per_fold_rows, columns=cols)
    pf.to_csv(os.path.join(model_dir, 'per_fold_results.csv'), index=False)
    pd.DataFrame(history_rows,
                 columns=['Fold', 'Epoch', 'TrainLoss', 'TrainAcc', 'ValLoss', 'ValAcc']
                 ).to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    np.savez_compressed(os.path.join(model_dir, 'oof_predictions.npz'),
                        y_true=np.concatenate(oof_true),
                        y_pred=np.concatenate(oof_pred),
                        y_prob=np.concatenate(oof_prob))
    if best_state is not None:
        torch.save(best_state, os.path.join(model_dir, f'{name}_best_overall.pth'))

    # ---- edge profile ----
    model = builder().to(device)
    if best_state is not None:
        model.load_state_dict(best_state)
    prof = profile_edge(model, model_dir, name)

    # ---- summary row (mean +/- std) ----
    summary = {'model': name, 'group': group}
    for metric in ['Accuracy', 'F1', 'Sensitivity', 'Precision', 'Specificity', 'MCC', 'AUC']:
        summary[f'{metric}_mean'] = pf[metric].mean()
        summary[f'{metric}_std'] = pf[metric].std(ddof=1)
    summary['TrainTime_s_mean'] = pf['TrainTime_s'].mean()
    summary.update({f'edge_{k}': v for k, v in prof.items()})
    print(f"  DONE {name}: acc={summary['Accuracy_mean']:.4f}"
          f"+/-{summary['Accuracy_std']:.4f}  params={prof['parameters_millions']}M")
    return summary


# =============================================================================
# CROSS-DATASET EXTERNAL VALIDATION (zero-shot; open + constrained protocols)
# =============================================================================
def match_external_class(ext_name, asdid_classes):
    """Map an external folder name to an ASDID class index via keywords.
    Returns (asdid_index, asdid_name) or (None, None) to skip (e.g. SDS)."""
    e = ext_name.lower()
    if 'sudden' in e or e.strip() == 'sds':
        return None, None
    if 'rust' in e:
        key = 'rust'
    elif 'bacterial' in e:
        key = 'bacterial'
    elif 'cercospora' in e:
        key = 'cercospora'
    elif 'healthy' in e:
        key = 'healthy'
    else:
        return None, None
    for i, c in enumerate(asdid_classes):
        if key in c.lower():
            return i, c
    return None, None


@torch.no_grad()
def cross_dataset_eval(asdid_classes):
    if not EXTERNAL_DATASET_PATH or not os.path.isdir(EXTERNAL_DATASET_PATH):
        print("Cross-dataset path not found; skipping external validation.")
        return
    model_dir = os.path.join(RESULTS_ROOT, PROPOSED_MODEL)
    ckpt = os.path.join(model_dir, f'{PROPOSED_MODEL}_best_overall.pth')
    if not os.path.exists(ckpt):
        print("Proposed-model checkpoint missing; run training first.")
        return

    print(f"\n{'='*70}\nCROSS-DATASET EXTERNAL VALIDATION\n{'='*70}")
    model = MODEL_REGISTRY[PROPOSED_MODEL][0]().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ext = datasets.ImageFolder(EXTERNAL_DATASET_PATH, transform=base_transform())
    # external folder index -> ASDID index (shared classes only)
    ext_to_asdid, shared = {}, {}
    for ext_name, ext_idx in ext.class_to_idx.items():
        a_idx, a_name = match_external_class(ext_name, asdid_classes)
        if a_idx is not None:
            ext_to_asdid[ext_idx] = a_idx
            shared[a_idx] = a_name
    shared_asdid_idx = sorted(shared.keys())
    print(f"  shared classes (ASDID idx -> name): {shared}")
    print(f"  excluded external classes (e.g. SDS) are dropped from the test")

    loader = DataLoader(ext, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    y_true, logits_all = [], []
    for inputs, ext_labels in tqdm(loader, desc="external inference"):
        keep = [i for i, l in enumerate(ext_labels.tolist()) if l in ext_to_asdid]
        if not keep:
            continue
        inputs = inputs[keep].to(device)
        out = model(inputs).cpu().numpy()
        for i, gi in enumerate(keep):
            y_true.append(ext_to_asdid[int(ext_labels[gi])])
            logits_all.append(out[i])
    y_true = np.array(y_true)
    logits_all = np.array(logits_all)

    # open protocol: argmax over all 8 classes
    y_pred_open = logits_all.argmax(1)
    # constrained protocol: argmax only over shared ASDID indices
    sub = logits_all[:, shared_asdid_idx]
    y_pred_con = np.array(shared_asdid_idx)[sub.argmax(1)]

    out_dir = os.path.join(RESULTS_ROOT, 'cross_dataset')
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'predictions.npz'),
                        y_true=y_true, y_pred_open=y_pred_open,
                        y_pred_constrained=y_pred_con,
                        shared_idx=np.array(shared_asdid_idx))

    def block(y_pred, tag):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', labels=shared_asdid_idx, zero_division=0)
        sen = recall_score(y_true, y_pred, average='macro', labels=shared_asdid_idx, zero_division=0)
        pre = precision_score(y_true, y_pred, average='macro', labels=shared_asdid_idx, zero_division=0)
        print(f"  [{tag}] acc={acc:.4f} f1={f1:.4f} sen={sen:.4f} pre={pre:.4f}")
        per_class = {}
        for idx in shared_asdid_idx:
            per_class[shared[idx]] = {
                'precision': precision_score(y_true, y_pred, labels=[idx], average='micro', zero_division=0),
                'recall':    recall_score(y_true, y_pred, labels=[idx], average='micro', zero_division=0),
                'support':   int((y_true == idx).sum()),
            }
        return {'accuracy': acc, 'f1_macro': f1, 'sensitivity_macro': sen,
                'precision_macro': pre, 'per_class': per_class}

    metrics = {'n_test_images': int(len(y_true)),
               'shared_classes': shared,
               'open': block(y_pred_open, 'open'),
               'constrained': block(y_pred_con, 'constrained')}
    with open(os.path.join(out_dir, 'cross_dataset_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved -> {out_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Load ORIGINAL dataset once, as PIL (transform=None). Transforms are
    # applied per-fold so augmentation never touches validation data.
    pil_dataset = datasets.ImageFolder(ORIGINAL_DATASET_PATH, transform=None)
    targets = np.array(pil_dataset.targets)
    asdid_classes = pil_dataset.classes
    print(f"Loaded {len(pil_dataset)} original images, classes: {asdid_classes}")
    with open(os.path.join(RESULTS_ROOT, 'class_names.json'), 'w') as f:
        json.dump({'classes': asdid_classes,
                   'class_to_idx': pil_dataset.class_to_idx}, f, indent=2)

    n_folds = 2 if QUICK_TEST else NUM_FOLDS
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(kfold.split(np.arange(len(pil_dataset))))

    summaries = []
    for name in MODELS_TO_RUN:
        summaries.append(run_model(name, pil_dataset, targets, splits))

    pd.DataFrame(summaries).to_csv(
        os.path.join(RESULTS_ROOT, 'all_models_summary.csv'), index=False)
    print(f"\nSummary written -> {os.path.join(RESULTS_ROOT, 'all_models_summary.csv')}")

    cross_dataset_eval(asdid_classes)
    print("\nALL DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='1-epoch/2-fold smoke test')
    parser.add_argument('--only', type=str, default=None,
                        help='comma-separated subset of model names to run')
    args = parser.parse_args()
    if args.quick:
        QUICK_TEST = True
    if args.only:
        MODELS_TO_RUN = [m.strip() for m in args.only.split(',')]
    main()
