# Chilli cross-dataset PyTorch experiment suite

Target study title:

**Cross-Dataset Generalization of Lightweight Deep Learning for Multi-Class Chilli Leaf Disease Classification**

## 1. Expected project structure

Place these scripts in the project root beside `data/`:

```text
Chili_CrossDataset/
├── config.yaml
├── common.py
├── data_pipeline.py
├── baseline_models.py
├── proposed_model.py
├── train_engine.py
├── experiment_runner.py
├── inspect_data.py
├── run_baselines.py
├── run_proposed.py
├── requirements.txt
└── data/
    ├── zip/
    ├── raw/
    │   ├── dataset_A_bangladesh/
    │   ├── dataset_B_field8814/
    │   └── dataset_C_cold_india/
    └── manifests/
```

Run `extract_data.py` first.

## 2. Install dependencies

For CUDA, install the correct PyTorch build using the official PyTorch
installation selector. Then install the remaining packages.

```bash
pip install -r requirements.txt
```

## 3. Inspect the actual data before training

```bash
python inspect_data.py --rebuild-index
```

This command:

- scans recursively;
- uses the six-class alias map in `config.yaml`;
- excludes repository-provided augmented paths;
- restricts Dataset C to the configured raw-image representation;
- computes SHA-256 hashes;
- drops exact duplicates inside each dataset;
- checks for exact-image label conflicts;
- prints class counts and shared class intersections.

**Do not start the full experiment until the printed class matrix is sensible.**

If Dataset C has a different extracted raw-folder name, edit:

```yaml
data:
  datasets:
    C:
      include_any:
        - resized raw images
```

## 4. Dry-run the experiment plan

```bash
python run_baselines.py --experiment pairwise --models mobilenet_v2 --dry-run
python run_proposed.py --experiment pairwise --dry-run
```

## 5. Run a small first experiment

One baseline, within-dataset 5-fold CV on Dataset A:

```bash
python run_baselines.py \
  --experiment within_cv \
  --models mobilenet_v2 \
  --datasets A
```

Proposed model under the same protocol:

```bash
python run_proposed.py \
  --experiment within_cv \
  --datasets A
```

## 6. Full baseline experiments

Within-dataset 5-fold CV:

```bash
python run_baselines.py --experiment within_cv
```

Bidirectional pairwise cross-dataset evaluation:

```bash
python run_baselines.py --experiment pairwise
```

Multi-source held-out evaluation:

```bash
python run_baselines.py --experiment multisource
```

Pooled multi-source 5-fold CV:

```bash
python run_baselines.py --experiment pooled_cv
```

Everything:

```bash
python run_baselines.py --experiment all
```

## 7. Full proposed-model experiments

```bash
python run_proposed.py --experiment within_cv
python run_proposed.py --experiment pairwise
python run_proposed.py --experiment multisource
python run_proposed.py --experiment pooled_cv
```

## 8. Final multi-seed runs

The default is one seed to keep development practical. For the final key
cross-dataset comparisons, edit `config.yaml`:

```yaml
experiments:
  pairwise:
    seeds: [42, 52, 62]

  multisource:
    seeds: [42, 52, 62]
```

Do not change the training hyperparameters between baselines and the proposed
model.

## 9. Fairness rules implemented

Both model families share:

- the same data index;
- the same exact train/validation/test split CSVs;
- the same 224×224 input size;
- the same train/eval transforms;
- the same ImageNet normalization;
- the same ImageNet-pretraining switch;
- the same AdamW optimizer;
- the same learning rate and weight decay;
- the same cosine schedule and warm-up;
- the same class-balanced cross-entropy;
- the same label smoothing;
- the same early-stopping rule;
- the same AMP setting;
- the same metrics;
- the same duplicate-overlap protection.

For cross-dataset tests, any byte-identical target image also found in the
source is removed before evaluation.

## 10. Important interpretation

- `within_cv`: internal, group-aware outer 5-fold CV with an inner validation
  split for early stopping. The outer fold is not used for model selection.
- `pairwise`: source-only training/validation, then frozen evaluation on the
  independent target dataset using shared classes only.
- `multisource`: source-union training, then held-out target-domain testing.
- `pooled_cv`: additional pooled analysis; it is **not** external validation.

## 11. Proposed model

`proposed_model.py` contains an editable prototype called
`ChiliLiteGFNet`. It currently uses:

- MobileNetV3-Small convolutional features;
- multi-scale taps;
- lightweight projections;
- ECA-style recalibration;
- adaptive gated fusion;
- depthwise-separable refinement.

The architecture is intentionally isolated. You can change it many times
without touching the shared experiment protocol.


## 12. Statistical testing

The training engine now saves, for every run:

```text
history.csv
predictions.csv
confusion_matrix.csv
per_class_metrics.csv
split_class_counts.csv
metrics.json
labels.json
best_model.pt
train_split.csv
val_split.csv
test_split.csv
```

Run-level statistical analysis:

```bash
python statistical_analysis.py \
  --experiment within_cv \
  --metric test_macro_f1
```

For pairwise cross-dataset experiments:

```bash
python statistical_analysis.py \
  --experiment pairwise \
  --metric test_macro_f1
```

This generates:

```text
results/statistics/<experiment>/
├── descriptive_summary.csv
├── average_ranks.csv
├── friedman_tests.csv
└── pairwise_wilcoxon_holm.csv
```

For the final proposed-vs-baseline comparison, first run multiple seeds in
`config.yaml`, for example:

```yaml
pairwise:
  seeds: [42, 52, 62, 72, 82]

multisource:
  seeds: [42, 52, 62, 72, 82]
```

Then run prediction-level paired tests:

```bash
python statistical_analysis.py \
  --experiment pairwise \
  --metric test_macro_f1 \
  --model-a chilli_lite_gfnet \
  --model-b mobilenet_v3_small \
  --bootstrap 2000
```

Additional outputs:

```text
mcnemar_exact__chilli_lite_gfnet__vs__mobilenet_v3_small.csv
paired_bootstrap__chilli_lite_gfnet__vs__mobilenet_v3_small.csv
```

Interpretation:

- Friedman: omnibus comparison across 3 or more models on matched units.
- Pairwise Wilcoxon: paired post-hoc comparison across matched units.
- Holm correction: controls family-wise error across pairwise model tests.
- Rank-biserial r: paired non-parametric effect size.
- Exact McNemar: paired disagreement test on the same test images.
- Paired bootstrap: 95% CI for the difference in a prediction-level metric.

Recommended manuscript emphasis:

- report mean ± SD across folds/seeds;
- report the Friedman omnibus p-value before pairwise post-hoc claims;
- report Holm-adjusted p-values;
- report effect size, not p-values alone;
- for the proposed model versus the strongest baseline, add exact McNemar and
  paired-bootstrap confidence intervals on identical test samples.

## Data-quality conflict handling

The data pipeline applies conservative audit rules before any split:

1. zero-byte files are excluded;
2. unreadable/corrupt images are excluded after PIL verification;
3. exact content with conflicting canonical labels is excluded in full;
4. same-label exact duplicates within a dataset are reduced to one deterministic representative;
5. source-target exact overlap is removed before cross-dataset evaluation.

Audit outputs are written to `data/manifests/`:

- `scan_log.csv`
- `data_quality_summary.csv`
- `conflicting_duplicate_labels.csv`
- `removed_exact_duplicates.csv`
- `dataset_class_counts.csv`
- `training_index.csv`

After replacing an older pipeline, rebuild the cache:

```bash
python3 inspect_data.py --rebuild-index
```

Then dry-run:

```bash
python3 run_baselines.py --experiment pairwise --models mobilenet_v2 --dry-run
```
