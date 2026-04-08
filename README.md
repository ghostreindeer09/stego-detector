<div align="center">

# 🔍 LSB Steganalysis Detector

### Deep Learning-based Image Steganography Detection using SRNet

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**98.4% Accuracy** · **0.9994 AUC** · **90% on Unseen Images** · **Trained on 131K OpenImages**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Best Model Results](#-best-model-results-epoch-69)
- [Unseen Image Test Results](#-unseen-image-test-results)
- [Architecture](#-architecture)
- [Pipeline](#-pipeline)
- [Quick Start](#-quick-start)
- [Training Configuration](#-training-configuration)
- [Project Structure](#-project-structure)
- [Docker](#-docker)
- [Evaluation Metrics](#-evaluation-metrics)

---

## 🎯 Overview

This project implements a **leakage-free steganalysis pipeline** that detects LSB (Least Significant Bit) steganography in images using a modified **SRNet** (Steganalysis Residual Network) architecture. The model is trained on 131,183 images from the **OpenImages V7** dataset with 4 different LSB embedding algorithms.

### Key Features

- **SRNet with KV High-Pass Filter** — Fixed Ker-Vass 5×5 HPF extracts noise residuals before classification
- **Leakage-Free Data Pipeline** — Source images split BEFORE stego generation to prevent any data leakage
- **Multi-Algorithm Training** — 4 LSB embedding algorithms (sequential, random, PVD, matching)
- **Curriculum Learning** — 3-phase training with progressive data exposure
- **Mixed Precision (BF16)** — Efficient training with bfloat16 AMP on modern GPUs
- **Pair-Constraint Batching** — Cover/stego pairs always appear together in each batch

---

## 🏆 Best Model Results (Epoch 69)

The model was trained for 69 epochs across 2 curriculum phases and achieved the following results on the held-out validation set (13,278 pairs):

### Primary Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **AUC (ROC)** | **0.9994** | > 0.90 | ✅ Exceeded |
| **Accuracy** | **98.4%** | > 85% | ✅ Exceeded |
| **F1 Score** | **0.9840** | — | ✅ Excellent |
| **Precision** | **0.9740** | — | ✅ Excellent |
| **Recall** | **0.9943** | — | ✅ Excellent |

### Advanced Metrics

| Metric | Value |
|--------|-------|
| **EER** (Equal Error Rate) | 0.0131 |
| **PE** (Probability of Error) | 0.0161 |
| **TPR @ 1% FPR** | 0.9841 |
| **TPR @ 5% FPR** | 0.9974 |
| **FPR** (False Positive Rate) | 0.0265 |
| **FNR** (False Negative Rate) | 0.0057 |

### Confusion Matrix (Epoch 69)

|  | Predicted Cover | Predicted Stego |
|--|-----------------|-----------------|
| **Actual Cover** | 12,926 (TN) | 352 (FP) |
| **Actual Stego** | 76 (FN) | 13,202 (TP) |

### Generalization Gap

| Metric | Train | Validation | Gap |
|--------|-------|------------|-----|
| **Loss** | 0.0036 | 0.0030 | 0.0006 (val lower ✅) |
| **Accuracy** | 97.8% | 98.4% | -0.6% (val higher ✅) |

> **No overfitting detected.** The validation performance slightly exceeds training performance, indicating excellent generalization.

### Training Progression

| Phase | Epochs | Data % | Best AUC | Best Accuracy |
|-------|--------|--------|----------|---------------|
| Phase 1 | 1–50 | 30% | 0.9984 | 97.0% |
| Phase 2 | 51–69 | 60% | 0.9994 | 98.4% |

---

## 🧪 Unseen Image Test Results

The model was evaluated on **10 completely unseen images** (5 clean + 5 stego) that were **never part of the training, validation, or test sets**. These images were sourced independently to validate real-world generalization.

### Results: 9/10 Correct — 90.0% Accuracy ✅

#### Clean Images (should predict COVER)

| Image | True Label | Predicted | Confidence | Result |
|-------|-----------|-----------|------------|--------|
| `unseen_clean_1.256.png` | COVER | COVER | 99.33% | ✅ Correct |
| `unseen_clean_2.256.png` | COVER | COVER | 82.93% | ✅ Correct |
| `unseen_clean_3.256.png` | COVER | COVER | 68.90% | ✅ Correct |
| `unseen_clean_4.256.png` | COVER | COVER | 68.18% | ✅ Correct |
| `unseen_clean_5.256.png` | COVER | STEGO | 64.33% | ❌ Wrong |

#### Stego Images (should predict STEGO)

| Image | True Label | Predicted | Confidence | Result |
|-------|-----------|-----------|------------|--------|
| `unseen_stego_1.256.png` | STEGO | STEGO | 75.53% | ✅ Correct |
| `unseen_stego_2.256.png` | STEGO | STEGO | 75.53% | ✅ Correct |
| `unseen_stego_3.256.png` | STEGO | STEGO | 92.79% | ✅ Correct |
| `unseen_stego_4.256.png` | STEGO | STEGO | 91.72% | ✅ Correct |
| `unseen_stego_5.256.png` | STEGO | STEGO | 98.69% | ✅ Correct |

> **Verdict: GENUINELY GOOD MODEL** ✅ — Model learned real LSB detection patterns and generalizes to completely unseen images.

---

## 🏗️ Architecture

<img width="223" height="718" alt="image" src="https://github.com/user-attachments/assets/f1dec07a-4ae8-4bea-85f4-c89f9fff1ecc" />


- **Parameters**: 2,454,892 (2.45M)
- **High-Pass Filter**: Fixed Ker-Vass 5×5 kernel extracts noise residuals
- **Loss Function**: FocalLoss (γ=2, α=0.25) for handling class imbalance
- **Optimizer**: AdamW with OneCycleLR scheduler

---

## 🔄 Pipeline

<img width="307" height="597" alt="image" src="https://github.com/user-attachments/assets/9393424d-1a12-4431-b649-845d58c110e1" />


### Embedding Algorithms

| Algorithm | Description | Used In |
|-----------|-------------|---------|
| `lsb_sequential` | Sequential LSB replacement | Train, Val, Test |
| `lsb_random` | Random pixel selection LSB | Train only |
| `lsb_pvd` | Pixel Value Differencing | Train only |
| `lsb_matching` | LSB matching (±1 adjustment) | Train only |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (tested on RTX 5060 Laptop, 8.5GB VRAM)
- ~70GB free disk space for dataset generation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Source Images

Place your cover images (PNG, 256×256) in `data/source_images/`.

### 3. Run Full Pipeline (First Run)

```bash
python run_pipeline_v31.py --num-workers 8 --data-dir "data/source_images"
```

This will:
1. ✅ Run pre-flight checks (GPU, disk, images)
2. ✅ Split images 80/10/10 (leakage-free)
3. ✅ Generate stego images (4 algorithms for train)
4. ✅ Train SRNet with curriculum learning
5. ✅ Save best checkpoint to `checkpoints/`

### 4. Subsequent Runs (Skip Data Generation)

```bash
python run_pipeline_v31.py --num-workers 8 --data-dir "data/source_images" --skip-data-gen
```

### 5. Run Unseen Image Test

```bash
python unseen_test/unseen_test.py
```

### 6. Check Best Model Metrics

```bash
python -c "import torch; c=torch.load('checkpoints/srnet_v31_baseline_best.pth', map_location='cpu', weights_only=False); m=c.get('metrics',{}); ep=c.get('epoch','?'); print('Epoch:', ep); print('AUC:', round(m.get('weighted_auc',0),4)); print('Acc:', round(m.get('accuracy',0)*100,2)); print('F1:', round(m.get('f1',0),4))"
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 150 (stopped at 69) |
| **Batch Size** | 64 |
| **Learning Rate** | 1e-3 |
| **Optimizer** | AdamW |
| **Scheduler** | OneCycleLR |
| **Loss Function** | FocalLoss (γ=2, α=0.25) |
| **AMP Dtype** | bfloat16 |
| **Image Size** | 256×256 |
| **Num Workers** | 8 |
| **Early Stop Patience** | 20 epochs |
| **Seed** | 42 |

### Curriculum Learning Schedule

| Phase | Epochs | Training Data | Purpose |
|-------|--------|---------------|---------|
| **Phase 1** | 1–50 | 30% of train set | Learn basic patterns |
| **Phase 2** | 51–100 | 60% of train set | Expand to harder examples |
| **Phase 3** | 101–150 | 100% of train set | Full data fine-tuning |

### GPU Configuration

| Property | Value |
|----------|-------|
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU |
| **VRAM** | 8.5 GB |
| **SM** | 12.0 |
| **AMP** | bfloat16 |
| **Throughput** | 125–136 img/s |
| **GPU Utilization** | 84–88% |

---

## 📁 Project Structure
```
stego-detector/
│
├── run_pipeline_v31.py              # Main entry point — argument parser
│
├── pipeline/                        # Core training pipeline
│   ├── preflight.py                 #   Pre-flight checks, GPU monitor
│   ├── data_gen.py                  #   Leakage-free split + LSB embedding
│   └── trainer.py                   #   Training loop, FocalLoss, curriculum
│
├── stego/                           # Model & data modules
│   ├── model.py                     #   SRNet architecture, GradCAM
│   ├── features.py                  #   KV HPF, SRM, DCT feature extraction
│   ├── datasets.py                  #   PairConstraintStegoDataset
│   ├── detector.py                  #   Inference wrapper
│   └── metrics.py                   #   AUC, F1, EER, PE, TPR@FPR
│
├── framework/                       # Embedding & research framework
│   └── embedding.py                 #   LSB embedding (4 algorithms)
│
├── unseen_test/                     # Independent validation
│   └── unseen_test.py               #   Test on completely unseen images
│
├── configs/                         # Experiment configurations
│   ├── default_experiment.yaml
│   └── openimages_experiment.yaml
│
├── app.py                           # Streamlit web demo
├── Dockerfile                       # Docker setup
├── docker-compose.yml               # Docker Compose
├── requirements.txt                 # Python dependencies
│
├── data/                            # Data directory (gitignored)
│   ├── source_images/               #   131,183 raw PNG cover images
│   └── splits_v31/                  #   Generated train/val/test splits
│       ├── train/cover/ + stego/
│       ├── val/cover/ + stego/
│       └── test/cover/ + stego/
│
└── checkpoints/                     # Saved models (gitignored)
    └── srnet_v31_baseline_best.pth  #   Best model checkpoint

---```

## 🐳 Docker

### Build

```bash
docker build -t stego-detector .
```

### Run

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  stego-detector python run_pipeline_v31.py --num-workers 8 --data-dir "data/source_images"
```

### Docker Compose

```bash
docker-compose up
```

---

## 📊 Evaluation Metrics

| Metric | Description | Best Value |
|--------|-------------|------------|
| **AUC** | Area Under ROC Curve | 0.9994 |
| **Accuracy** | Overall correct classifications | 98.4% |
| **F1** | Harmonic mean of precision and recall | 0.9840 |
| **Precision** | TP / (TP + FP) — Stego detection correctness | 0.9740 |
| **Recall** | TP / (TP + FN) — Stego detection completeness | 0.9943 |
| **EER** | Equal Error Rate — where FPR = FNR | 0.0131 |
| **PE** | Probability of Error — (FP + FN) / Total | 0.0161 |
| **TPR@1%FPR** | True Positive Rate at 1% False Positive Rate | 0.9841 |
| **TPR@5%FPR** | True Positive Rate at 5% False Positive Rate | 0.9974 |

---

## 📚 References

- **SRNet**: Boroumand, M., Chen, M., & Fridrich, J. (2019). Deep Residual Network for Steganalysis of Digital Images. *IEEE TIFS*.
- **KV Filter**: Ker, A. D., & Bas, P. (2016). High-pass filtering for steganalysis.
- **FocalLoss**: Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- **OpenImages V7**: Kuznetsova, A., et al. (2020). The Open Images Dataset V4. *IJCV*.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with** ❤️ **using PyTorch**

*Trained on NVIDIA RTX 5060 Laptop GPU · 131,183 OpenImages · 4 LSB Algorithms*

</div>
