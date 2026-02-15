"""
Config loader — reads YAML experiment configs and provides typed access.
"""

import os
import copy
import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*; override values win."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class ExperimentConfig:
    """Typed wrapper around a raw YAML config dict."""

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg

    # ---- Top-level sections ----
    @property
    def experiment(self) -> dict:
        return self._cfg.get("experiment", {})

    @property
    def dataset(self) -> dict:
        return self._cfg.get("dataset", {})

    @property
    def dataloader(self) -> dict:
        return self._cfg.get("dataloader", {})

    @property
    def model(self) -> dict:
        return self._cfg.get("model", {})

    @property
    def training(self) -> dict:
        return self._cfg.get("training", {})

    @property
    def evaluation(self) -> dict:
        return self._cfg.get("evaluation", {})

    @property
    def robustness(self) -> dict:
        return self._cfg.get("robustness", {})

    @property
    def interpretability(self) -> dict:
        return self._cfg.get("interpretability", {})

    @property
    def tracking(self) -> dict:
        return self._cfg.get("tracking", {})

    # ---- Convenience ----
    @property
    def seed(self) -> int:
        return self.experiment.get("seed", 42)

    @property
    def name(self) -> str:
        return self.experiment.get("name", "unnamed")

    @property
    def output_dir(self) -> str:
        return self.experiment.get("output_dir", "outputs")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Access nested keys with dot notation, e.g. 'training.epochs'."""
        keys = dotted_key.split(".")
        obj = self._cfg
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)

    def __repr__(self) -> str:
        return f"ExperimentConfig(name={self.name!r})"


def load_config(
    config_path: str,
    override_path: Optional[str] = None,
) -> ExperimentConfig:
    """
    Load a YAML config file; optionally merge an override file on top.

    Parameters
    ----------
    config_path : str
        Path to the base YAML config.
    override_path : str, optional
        Path to an override YAML that is deep-merged over the base.

    Returns
    -------
    ExperimentConfig
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if override_path:
        if not os.path.isfile(override_path):
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}
        base_cfg = _deep_merge(base_cfg, override_cfg)

    logger.info("Loaded config from %s", config_path)
    return ExperimentConfig(base_cfg)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Save an ExperimentConfig to a YAML file for reproducibility."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config to %s", path)
