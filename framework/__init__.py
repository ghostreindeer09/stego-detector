"""
Modular Steganalysis Framework
================================
Production-quality, config-driven framework for steganalysis research.

Modules:
    - config          : YAML configuration loading and validation
    - dataset_generator: Automated stego dataset creation with CSV mapping
    - dataset_loader   : Unified dataset loading for training / evaluation
    - embedding        : Embedding simulators (LSB, etc.)
    - feature_extractor: Classical ML feature extraction
    - trainer          : Model training orchestration
    - evaluator        : Comprehensive metrics evaluation
    - robustness       : Robustness testing under perturbations
    - interpretability : Grad-CAM, SHAP analysis (optional)
    - plotting         : Publication-quality plots
    - tracking         : MLflow / W&B experiment tracking (optional)
"""

__version__ = "1.0.0"
