# Modular Steganalysis Research Framework

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)


A production-quality, config-driven framework for steganographic image analysis research. Supports automated dataset creation, CNN and classical ML training, comprehensive evaluation, robustness testing, and optional interpretability — all from a single YAML config.

---

## 🏗️ Architecture

```
Pro/
├── stego/                          # Core model modules (PRESERVED — existing SRNet pipeline)
│   ├── model.py                    #   SRNet, SRNetBackbone, GradCAM
│   ├── features.py                 #   HPF, SRM, DCT feature extraction
│   ├── datasets.py                 #   Pair-constraint dataset classes
│   ├── detector.py                 #   StegoDetector (inference + Grad-CAM)
│   └── metrics.py                  #   Weighted AUC, F1, confusion matrix
│
├── framework/                      # NEW — Modular research framework
│   ├── config.py                   #   YAML config loader with deep-merge
│   ├── embedding.py                #   LSB embedder + extensible base class
│   ├── dataset_generator.py        #   Automated stego dataset creation
│   ├── dataset_loader.py           #   CSV-based DataLoader factory
│   ├── feature_extractor.py        #   Classical ML features (histogram, GLCM, DCT, etc.)
│   ├── trainer.py                  #   CNNTrainer + ClassicalMLTrainer
│   ├── evaluator.py                #   Comprehensive metrics + comparison tables
│   ├── robustness.py               #   Perturbation sweep testing
│   ├── interpretability.py         #   Grad-CAM + SHAP analysis
│   ├── plotting.py                 #   Publication-quality plots
│   └── tracking.py                 #   MLflow / W&B experiment tracking
│
├── configs/                        # Experiment configurations
│   ├── default_experiment.yaml     #   Full baseline config
│   └── robustness_sweep.yaml       #   Robustness-focused sweep config
│
├── train.py                        # PRESERVED — Original SRNet training script
├── train_alaska2.py                # PRESERVED — Original ALASKA2 training
├── evaluate_confusion_matrix.py    # PRESERVED — Original evaluation
├── app.py                          # PRESERVED — Streamlit demo app
│
├── generate_dataset.py             # NEW — Dataset generation CLI
├── run_experiment.py               # NEW — Main experiment runner
│
├── Dockerfile                      # NEW — Docker setup
├── docker-compose.yml              # NEW — Docker Compose
├── requirements.txt                # Updated dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate a Stego Dataset

Place your cover images in `cover_images/`, then run:

```bash
# Using CLI arguments:
python generate_dataset.py --cover-dir cover_images/ --stego-dir stego_images/

# Or using YAML config:
python generate_dataset.py --config configs/default_experiment.yaml
```

This creates:
- `stego_images/` — stego versions of each cover image
- `dataset_mapping.csv` — CSV with `image_name, label, payload, payload_length, source_path`

### 3. Run an Experiment

```bash
python run_experiment.py --config configs/default_experiment.yaml
```

This will:
1. ✅ Generate dataset (if CSV doesn't exist)
2. ✅ Load and split data (train/val/test)
3. ✅ Train SRNet with AMP, label smoothing, early stopping
4. ✅ Evaluate on test set (precision, recall, F1, AUC, confusion matrix)
5. ✅ Save plots and metrics to `outputs/`

### 4. Skip Steps

```bash
# Skip dataset generation (use existing CSV):
python run_experiment.py --config configs/default_experiment.yaml --skip-generation

# Skip training (evaluate only with existing checkpoint):
python run_experiment.py --config configs/default_experiment.yaml --skip-training

# Skip robustness testing:
python run_experiment.py --config configs/default_experiment.yaml --skip-robustness
```

---

## 🐳 Docker

```bash
# Build
docker build -t steganalysis .

# Run experiment
docker run -v $(pwd)/cover_images:/app/cover_images \
           -v $(pwd)/outputs:/app/outputs \
           steganalysis python run_experiment.py --config configs/default_experiment.yaml

# Or use Docker Compose
docker-compose up
```

---

## ⚙️ Configuration (YAML)

All experiment parameters live in a single YAML file. See `configs/default_experiment.yaml` for the full reference.

Key sections:

| Section | Controls |
|---------|----------|
| `experiment` | Name, seed, output directory |
| `dataset` | Cover/stego dirs, embedding method, payload settings |
| `dataloader` | Batch size, splits, augmentations |
| `model` | Architecture (srnet/classical_ml), hyperparameters |
| `training` | Epochs, optimizer, scheduler, AMP, early stopping |
| `evaluation` | Threshold, which metrics to compute, which plots |
| `robustness` | JPEG/noise/resize/crop/payload perturbation sweeps |
| `interpretability` | Grad-CAM and SHAP settings |
| `tracking` | MLflow / W&B integration |

### Override Configs

```bash
python run_experiment.py --config configs/default_experiment.yaml \
                         --override configs/robustness_sweep.yaml
```

---

## 📊 Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) — How many detected stego are correct |
| **Recall** | TP / (TP + FN) — How many stego images are found |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **Accuracy** | Overall correct classifications |
| **Specificity** | TN / (TN + FP) — True negative rate |
| **FP/FN Counts** | Raw false positive and false negative counts |

---

## 🔬 Robustness Testing

Test model resilience against real-world perturbations:

```yaml
robustness:
  perturbations:
    jpeg_compression:
      qualities: [50, 70, 80, 90, 95]
    gaussian_noise:
      sigmas: [1.0, 5.0, 10.0]
    resize:
      scales: [0.5, 0.75, 1.5]
    crop:
      ratios: [0.5, 0.7, 0.9]
    payload_size:
      lengths: [4, 16, 64, 128]
```

---

## 🧪 Research Experiments (TODOs)

The framework includes clear TODO markers for extending:

1. **Payload size robustness** — Re-embed with different payload lengths and measure detection rate
2. **Classical ML pipeline** — Extract features from DataLoader and train sklearn models
3. **Model comparison** — Run multiple architectures and use `plot_model_comparison()`
4. **SHAP analysis** — Install `shap` and use `SHAPAnalyzer` for feature importance
5. **Custom embedding methods** — Subclass `BaseEmbedder` in `framework/embedding.py`
6. **Cross-dataset transfer** — Train on ALASKA2, test on BOSSBase

---

## 📈 Optional: Experiment Tracking

### MLflow
```bash
pip install mlflow
```
Set in config:
```yaml
tracking:
  enabled: true
  backend: "mlflow"
```

### Weights & Biases
```bash
pip install wandb
```
```yaml
tracking:
  enabled: true
  backend: "wandb"
  wandb:
    project: "steganalysis"
```

---

## 🔄 Preserving Existing Pipeline

All original scripts are **fully preserved and functional**:

- `train.py` — Original SRNet training with pair-constraint batches
- `train_alaska2.py` — ALASKA2-specific training loop
- `evaluate_confusion_matrix.py` — Standalone confusion matrix evaluation
- `app.py` — Streamlit web demo

The new `framework/` module is entirely additive and does not modify any existing code.

---

## 📁 Output Structure

After running an experiment:

```
outputs/
├── resolved_config.yaml        # Exact config used (for reproducibility)
├── test_metrics.json           # Full test set metrics
├── plots/
│   ├── loss_curve.png
│   ├── auc_curves.png
│   ├── f1_curves.png
│   └── lr_schedule.png
├── evaluation_results/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
└── robustness_results/
    ├── robustness_results.json
    ├── robustness_jpeg_compression.png
    └── robustness_gaussian_noise.png
```
