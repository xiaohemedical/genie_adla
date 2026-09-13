# Genie-ADLA

Genie-ADLA is a deep-learning framework for binary classification from
high-dimensional numerical features. Developed for DNA methylation
classification experiments, it combines piecewise-linear numerical embeddings
with parameter-efficient ensemble learning.

The repository provides a complete training and evaluation workflow with two
execution modes:

- a single stratified train/validation split;
- stratified K-fold cross-validation with out-of-fold (OOF) prediction.

## Model architecture

```text
Numerical features                    (B, N)
  → Piecewise Linear Embedding        (B, N, E)
  → flatten                           (B, N × E)
  → K ensemble views                  (B, K, N × E)
  → parameter-efficient blocks        (B, K, H)
  → classification heads              (B, K, 2)
```

Each ensemble block uses a shared weight matrix together with member-specific
input and output scales:

```text
y = ((x * r) @ Wᵀ) * s + b
```

The final positive-class probability is the mean of the member-level softmax
probabilities.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

## Data format

Train and test data are CSV files with one sample per row:

```text
sample_id,label,feature_001,feature_002,...
sample_00001,0,0.13,0.82,...
sample_00002,1,0.76,0.31,...
```

The input data must satisfy the following requirements:

- labels are encoded as `0` and `1`;
- all features are numerical and finite;
- sample IDs are present and unique within each file;
- train and test feature names and order match exactly.

Constant features are not supported and must be removed before training. Small
simulated datasets are provided in `examples/data/`; they contain no patient or
methylation data.

## Configuration and training

All settings are defined in `configs/demo.yaml`, including data paths,
preprocessing, model dimensions, optimization, evaluation, and output paths.
The bundled values are lightweight defaults intended for the simulated data.

Run training after installation:

```bash
genie-adla-train --config configs/demo.yaml
```

Alternatively, run directly from the source tree:

```bash
PYTHONPATH=src python -m genie_adla.cli --config configs/demo.yaml
```

By default, training uses one stratified train/validation split. Enable K-fold
cross-validation in the configuration:

```yaml
cross_validation:
  enabled: true
  n_splits: 5
  shuffle: true
  split_seed: 42
```

In cross-validation mode, each fold fits its own preprocessor, numerical bins,
and model using only that fold's training partition. Each training sample
receives one OOF probability, while test probabilities are averaged across all
fold models:

```text
fold training partitions
  → fold-specific preprocessing, bin estimation, and model fitting
  → fold validation predictions
  → merged OOF predictions over the complete training cohort

fold test predictions
  → mean probability across fold models
  → final test prediction
```

## Threshold-based evaluation

`evaluation.target_specificities` specifies the requested specificity levels.
`evaluation.threshold_source` determines which negative-class scores are used
to estimate the corresponding decision thresholds:

| Run mode | `validation` | `test` |
| --- | --- | --- |
| Single split | Validation-set scores | Test-set scores |
| Cross-validation | OOF scores from the complete training cohort | Test-set scores |

The selected thresholds are applied to the test predictions to report
sensitivity and the achieved specificity. Using `threshold_source: test` is a
test-set-conditioned evaluation protocol; `validation` keeps threshold
selection independent of the test set.

## Outputs

Each run creates a timestamped directory under `output.directory`.

Single-split run:

```text
outputs/run_<timestamp>/
├── config.yaml
├── metadata.json
├── metrics.json
├── model.pt
├── preprocessor.joblib
├── validation_predictions.csv
└── test_predictions.csv
```

Cross-validation run:

```text
outputs/run_<timestamp>/
├── config.yaml
├── metadata.json
├── metrics.json
├── oof_predictions.csv
├── test_predictions.csv
├── fold_01/
│   ├── model.pt
│   ├── preprocessor.joblib
│   └── validation_predictions.csv
├── fold_02/
│   └── ...
└── fold_05/
    └── ...
```

The saved metadata records the feature order, software versions, execution
mode, device, and threshold source. Each `model.pt` must be used with its paired
preprocessor. Models can be restored with `genie_adla.load_model()`.

## Data availability

Real DNA methylation cohorts, study-specific configurations, trained
checkpoints, and research outputs are not distributed due to data-governance
and confidentiality requirements.

## License

Genie-ADLA is distributed under the [Apache License 2.0](LICENSE).

## References and acknowledgements

- *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*,
  ICLR 2025. [Paper](https://openreview.net/forum?id=SdF7Qn3Rrz) and
  [official repository](https://github.com/yandex-research/tabm). Copyright 2024
  Yandex LLC; licensed under Apache License 2.0.
- *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS 2022.
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)
  and [official repository](https://github.com/yandex-research/rtdl-num-embeddings).
  Copyright (c) 2022 the paper authors; licensed under the MIT License.

The referenced repositories are not vendored and are not runtime dependencies.
Genie-ADLA is not affiliated with or endorsed by their authors or maintainers.
