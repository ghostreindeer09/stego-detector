#!/usr/bin/env python3
"""
run_experiment.py
==================
Main entry point for config-driven steganalysis experiments.

Orchestrates:
    1. (Optional) Dataset generation
    2. Data loading
    3. Model training (CNN or Classical ML)
    4. Evaluation with full metrics
    5. (Optional) Robustness testing
    6. (Optional) Interpretability analysis
    7. Plotting and result export

Usage:
    python run_experiment.py --config configs/default_experiment.yaml
    python run_experiment.py --config configs/default_experiment.yaml --skip-training
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.config import load_config, save_config
from framework.tracking import build_tracker


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run steganalysis experiment.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--override", type=str, default=None, help="Override YAML.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    logger = logging.getLogger("experiment")

    # 1. Load config
    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config for reproducibility
    save_config(cfg, os.path.join(output_dir, "resolved_config.yaml"))
    logger.info("Experiment: %s", cfg.name)

    # 2. Tracker
    tracker = build_tracker(cfg)
    if tracker:
        tracker.start_run(cfg.name, cfg.to_dict())

    # 3. Dataset generation
    if not args.skip_generation:
        csv_path = cfg.dataset.get("mapping_csv", "dataset_mapping.csv")
        if not os.path.isfile(csv_path):
            logger.info("=== Phase: Dataset Generation ===")
            from framework.dataset_generator import generate_from_config
            generate_from_config(cfg)
        else:
            logger.info("CSV mapping already exists at '%s', skipping generation.", csv_path)

    # 4. Load data
    logger.info("=== Phase: Data Loading ===")
    from stego.datasets import get_train_transform, get_val_transform
    img_size = cfg.dataloader.get("image_size", 256)
    train_tf = get_train_transform(image_size=img_size)
    val_tf = get_val_transform(image_size=img_size)

    from framework.dataset_loader import build_dataloaders
    loaders = build_dataloaders(cfg, train_transform=train_tf, val_transform=val_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 5. Build model
    arch = cfg.model.get("architecture", "srnet")
    results_all = {}

    if arch in ("srnet", "custom_cnn"):
        logger.info("=== Phase: CNN Training / Evaluation ===")
        from stego.model import SRNet
        from framework.trainer import CNNTrainer
        from framework.evaluator import evaluate_model, print_metrics, save_metrics

        model = SRNet(
            num_classes=cfg.model.get("num_classes", 1),
            use_kv_hpf=cfg.model.get("use_kv_hpf", True),
        ).to(device)

        # Load pretrained if specified
        weights = cfg.model.get("pretrained_weights")
        if weights and os.path.isfile(weights):
            model.load_state_dict(torch.load(weights, map_location=device))
            logger.info("Loaded pretrained weights from %s", weights)

        # Train
        if not args.skip_training and cfg.training.get("epochs", 0) > 0:
            trainer = CNNTrainer(model, cfg, device)
            history = trainer.fit(loaders["train"], loaders["val"], tracker=tracker)

            # Plot training curves
            from framework.plotting import plot_training_curves
            plot_training_curves(history, save_dir=os.path.join(output_dir, "plots"))

        # Evaluate on test set
        threshold = cfg.evaluation.get("threshold", 0.5)
        test_metrics = evaluate_model(model, loaders["test"], device, threshold=threshold)
        print_metrics(test_metrics, title=f"Test Results — {cfg.name}")
        save_metrics(test_metrics, os.path.join(output_dir, "test_metrics.json"))
        results_all["SRNet"] = test_metrics

        # Plots
        eval_cfg = cfg.evaluation.get("plots", {})
        eval_dir = os.path.join(output_dir, cfg.evaluation.get("output_dir", "evaluation_results"))
        os.makedirs(eval_dir, exist_ok=True)

    if arch == "classical_ml":
        logger.info("=== Phase: Classical ML Training / Evaluation ===")
        from framework.trainer import ClassicalMLTrainer
        from framework.feature_extractor import extract_features
        from framework.evaluator import evaluate_classical_model, print_metrics, save_metrics
        from PIL import Image

        ml_trainer = ClassicalMLTrainer(cfg)
        feature_names = cfg.model.get("classical_ml", {}).get(
            "features", ["histogram", "cooccurrence", "dct_stats"]
        )

        logger.info("Extracting features for classical ML...")
        # TODO: Extract features from the dataset loader
        # This is a skeleton — fill in with actual feature extraction loop
        logger.info("TODO: Implement batch feature extraction from DataLoader for classical ML.")

    # 6. Robustness testing
    if cfg.robustness.get("enabled", False) and not args.skip_robustness:
        logger.info("=== Phase: Robustness Testing ===")
        # TODO: Load test images and run robustness tester
        logger.info(
            "TODO: Connect robustness tester to model predict function. "
            "See framework/robustness.py for the RobustnessTester class."
        )

    # 7. Interpretability
    if cfg.interpretability.get("enabled", False):
        logger.info("=== Phase: Interpretability ===")
        logger.info(
            "TODO: Run Grad-CAM / SHAP analysis. "
            "See framework/interpretability.py for GradCAMAnalyzer and SHAPAnalyzer."
        )

    # 8. Finish
    if tracker:
        tracker.end_run()

    logger.info("Experiment '%s' complete. Results in '%s'.", cfg.name, output_dir)


if __name__ == "__main__":
    main()
