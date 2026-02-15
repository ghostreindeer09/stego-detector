"""
Experiment Tracking (Optional)
===============================
Lightweight wrappers for MLflow and Weights & Biases.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseTracker:
    """Abstract tracker interface."""

    def start_run(self, run_name: str, config: dict) -> None:
        pass

    def log_epoch(self, epoch, train_loss, train_metrics, val_metrics, lr) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def end_run(self) -> None:
        pass


class MLflowTracker(BaseTracker):
    def __init__(self, tracking_uri: str = "mlruns", experiment_name: str = "steganalysis"):
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._available = True
        except ImportError:
            logger.warning("MLflow not installed. pip install mlflow")
            self._available = False

    def start_run(self, run_name: str, config: dict) -> None:
        if not self._available:
            return
        self.mlflow.start_run(run_name=run_name)
        self.mlflow.log_params(_flatten_dict(config))

    def log_epoch(self, epoch, train_loss, train_metrics, val_metrics, lr) -> None:
        if not self._available:
            return
        self.mlflow.log_metrics({
            "train_loss": train_loss,
            "train_auc": train_metrics.get("weighted_auc", 0),
            "train_f1": train_metrics.get("f1", 0),
            "val_auc": val_metrics.get("weighted_auc", 0),
            "val_f1": val_metrics.get("f1", 0),
            "val_precision": val_metrics.get("precision", 0),
            "val_recall": val_metrics.get("recall", 0),
            "lr": lr,
        }, step=epoch)

    def log_artifact(self, path: str) -> None:
        if self._available and os.path.exists(path):
            self.mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self._available:
            self.mlflow.end_run()


class WandBTracker(BaseTracker):
    def __init__(self, project: str = "steganalysis", entity: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            self._project = project
            self._entity = entity
            self._available = True
        except ImportError:
            logger.warning("W&B not installed. pip install wandb")
            self._available = False

    def start_run(self, run_name: str, config: dict) -> None:
        if not self._available:
            return
        self.wandb.init(project=self._project, entity=self._entity, name=run_name, config=config)

    def log_epoch(self, epoch, train_loss, train_metrics, val_metrics, lr) -> None:
        if not self._available:
            return
        self.wandb.log({
            "epoch": epoch, "train_loss": train_loss,
            "train_auc": train_metrics.get("weighted_auc", 0),
            "val_auc": val_metrics.get("weighted_auc", 0),
            "val_f1": val_metrics.get("f1", 0),
            "lr": lr,
        })

    def end_run(self) -> None:
        if self._available:
            self.wandb.finish()


def build_tracker(cfg) -> Optional[BaseTracker]:
    """Build a tracker from config; returns None if tracking disabled."""
    t = cfg.tracking
    if not t.get("enabled", False):
        return None
    backend = t.get("backend", "mlflow")
    if backend == "mlflow":
        mc = t.get("mlflow", {})
        return MLflowTracker(mc.get("tracking_uri", "mlruns"), mc.get("experiment_name", "steganalysis"))
    elif backend == "wandb":
        wc = t.get("wandb", {})
        return WandBTracker(wc.get("project", "steganalysis"), wc.get("entity"))
    return None


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
