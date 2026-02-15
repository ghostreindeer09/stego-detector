#!/usr/bin/env python3
"""
Smoke test: verifies that all framework modules import correctly
and the dataset generation pipeline works end-to-end.
"""

import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_imports():
    """Test that all framework modules import without error."""
    print("Testing imports...")
    from framework.config import load_config, ExperimentConfig
    from framework.embedding import LSBEmbedder, generate_random_message, get_embedder
    from framework.dataset_generator import generate_stego_dataset
    from framework.dataset_loader import CSVStegoDataset, split_dataset
    from framework.feature_extractor import extract_features, extract_histogram_features
    from framework.trainer import CNNTrainer, ClassicalMLTrainer, bce_with_label_smoothing
    from framework.evaluator import compute_all_metrics, print_metrics
    from framework.robustness import apply_jpeg_compression, apply_gaussian_noise
    from framework.plotting import plot_confusion_matrix
    from framework.tracking import build_tracker, BaseTracker
    from framework.interpretability import GradCAMAnalyzer

    # Also verify existing modules still work
    from stego.model import SRNet, SRNetBackbone, GradCAM
    from stego.features import FeatureExtractor, KVHighPassFilter
    from stego.detector import StegoDetector
    from stego.datasets import PairConstraintStegoDataset
    from stego.metrics import compute_metrics

    print("  ✓ All imports successful!")


def test_embedding():
    """Test LSB embedding and extraction."""
    from PIL import Image
    import numpy as np
    from framework.embedding import LSBEmbedder, generate_random_message

    print("Testing embedding...")
    embedder = LSBEmbedder()
    msg = generate_random_message(16, seed=42)

    img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    stego = embedder.embed(img, msg)
    extracted = embedder.extract(stego)

    assert extracted == msg, f"Mismatch: '{extracted}' != '{msg}'"
    print(f"  ✓ Embedded and extracted: '{msg}'")


def test_dataset_generation():
    """Test full dataset generation pipeline."""
    from PIL import Image
    import numpy as np
    import csv

    print("Testing dataset generation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cover_dir = os.path.join(tmpdir, "covers")
        stego_dir = os.path.join(tmpdir, "stegos")
        csv_path = os.path.join(tmpdir, "mapping.csv")
        os.makedirs(cover_dir)

        # Create dummy cover images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(cover_dir, f"img_{i:04d}.png"))

        from framework.dataset_generator import generate_stego_dataset
        count = generate_stego_dataset(
            cover_dir=cover_dir,
            stego_dir=stego_dir,
            mapping_csv=csv_path,
            seed=42,
        )

        assert count == 5, f"Expected 5 stego images, got {count}"
        assert os.path.isfile(csv_path), "CSV not created"

        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10, f"Expected 10 CSV rows, got {len(rows)}"

        covers = [r for r in rows if r["label"] == "0"]
        stegos = [r for r in rows if r["label"] == "1"]
        assert len(covers) == 5 and len(stegos) == 5

        print(f"  ✓ Generated {count} stego images, {len(rows)} CSV rows")


def test_config():
    """Test config loading."""
    print("Testing config...")
    from framework.config import load_config

    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default_experiment.yaml"))
    assert cfg.name == "baseline_srnet_lsb"
    assert cfg.seed == 42
    assert cfg.get("training.epochs") == 50
    print(f"  ✓ Config loaded: {cfg}")


def test_metrics():
    """Test metrics computation."""
    print("Testing metrics...")
    import numpy as np
    from framework.evaluator import compute_all_metrics

    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2, 0.6, 0.4])

    metrics = compute_all_metrics(y_true, y_scores, threshold=0.5)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(y_true)
    print(f"  ✓ Metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
          f"F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")


def test_perturbations():
    """Test perturbation functions."""
    from PIL import Image
    import numpy as np
    from framework.robustness import (
        apply_jpeg_compression, apply_gaussian_noise,
        apply_resize, apply_crop,
    )

    print("Testing perturbations...")
    img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))

    jpeg_img = apply_jpeg_compression(img, 50)
    assert jpeg_img.size == img.size

    noisy_img = apply_gaussian_noise(img, 5.0)
    assert noisy_img.size == img.size

    resized_img = apply_resize(img, 0.5)
    assert resized_img.size == img.size

    cropped_img = apply_crop(img, 0.7)
    assert cropped_img.size == img.size

    print("  ✓ All perturbations applied successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("  Steganalysis Framework — Smoke Tests")
    print("=" * 60)

    test_imports()
    test_embedding()
    test_dataset_generation()
    test_config()
    test_metrics()
    test_perturbations()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
