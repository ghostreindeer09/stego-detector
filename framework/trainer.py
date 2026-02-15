"""
Model Trainer
==============
Orchestrates training for both CNN (SRNet) and classical ML models.
Supports config-driven experiments, early stopping, AMP, and experiment tracking.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label-smoothed BCE (reuses existing logic)
# ---------------------------------------------------------------------------

def bce_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(1), targets_smooth.squeeze(1),
    )


# ---------------------------------------------------------------------------
# CNN Trainer
# ---------------------------------------------------------------------------

class CNNTrainer:
    """
    Config-driven CNN trainer wrapping the existing SRNet training loop.

    Parameters
    ----------
    model : nn.Module
    cfg   : ExperimentConfig
    device : torch.device
    """

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        t = cfg.training
        self.epochs = t.get("epochs", 50)
        self.lr = t.get("learning_rate", 1e-3)
        self.weight_decay = t.get("weight_decay", 1e-5)
        self.label_smoothing = t.get("label_smoothing", 0.1)
        self.use_amp = t.get("amp", {}).get("enabled", True) and device.type == "cuda"
        self.checkpoint_dir = t.get("checkpoint_dir", "checkpoints")
        self.save_name = t.get("save_name", "srnet_best.pth")

        # Early stopping
        es = t.get("early_stopping", {})
        self.early_stopping = es.get("enabled", False)
        self.patience = es.get("patience", 10)
        self.es_metric = es.get("metric", "weighted_auc")
        self.es_mode = es.get("mode", "max")

        # Optimizer
        opt_name = t.get("optimizer", "adamw").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        self.scaler = GradScaler() if self.use_amp else None

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_auc": [],
            "train_f1": [],
            "val_auc": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "lr": [],
        }

    def _build_scheduler(self, steps_per_epoch: int):
        sched_name = self.cfg.training.get("scheduler", "onecycle").lower()
        params = self.cfg.training.get("scheduler_params", {})

        if sched_name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=params.get("pct_start", 0.1),
            )
        elif sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs,
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.1),
            )
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5,
            )
        else:
            return None

    def train_one_epoch(
        self, loader: DataLoader, scheduler=None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train()
        running_loss = 0.0
        all_scores, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler is not None:
                with autocast():
                    logits, _ = self.model(images)
                    loss = bce_with_label_smoothing(logits, labels, self.label_smoothing)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(images)
                loss = bce_with_label_smoothing(logits, labels, self.label_smoothing)
                loss.backward()
                self.optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step()

            running_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels.squeeze(1).cpu().numpy())

        mean_loss = running_loss / max(1, len(loader))
        return mean_loss, np.concatenate(all_scores), np.concatenate(all_labels)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        from stego.metrics import compute_metrics

        self.model.eval()
        all_scores, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            if self.use_amp:
                with autocast():
                    logits, _ = self.model(images)
            else:
                logits, _ = self.model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

        return compute_metrics(
            np.concatenate(all_labels),
            np.concatenate(all_scores),
            threshold=self.cfg.evaluation.get("threshold", 0.5),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tracker=None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with optional experiment tracking.

        Returns training history dict.
        """
        from stego.metrics import compute_metrics

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        scheduler = self._build_scheduler(len(train_loader))
        best_metric = -float("inf") if self.es_mode == "max" else float("inf")
        patience_counter = 0

        logger.info("Starting training: %d epochs, device=%s", self.epochs, self.device)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_scores, train_labels = self.train_one_epoch(
                train_loader, scheduler,
            )
            train_metrics = compute_metrics(train_labels, train_scores)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_auc"].append(train_metrics["weighted_auc"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_auc"].append(val_metrics["weighted_auc"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %d/%d [%.1fs] | Loss=%.4f | Train AUC=%.4f F1=%.4f | "
                "Val AUC=%.4f F1=%.4f P=%.4f R=%.4f | FP=%d FN=%d",
                epoch, self.epochs, elapsed,
                train_loss,
                train_metrics["weighted_auc"], train_metrics["f1"],
                val_metrics["weighted_auc"], val_metrics["f1"],
                val_metrics["precision"], val_metrics["recall"],
                val_metrics["fp"], val_metrics["fn"],
            )

            # ReduceLROnPlateau step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[self.es_metric])

            # Experiment tracking
            if tracker is not None:
                tracker.log_epoch(epoch, train_loss, train_metrics, val_metrics, current_lr)

            # Checkpoint
            current_metric = val_metrics.get(self.es_metric, val_metrics["weighted_auc"])
            improved = (
                (self.es_mode == "max" and current_metric > best_metric)
                or (self.es_mode == "min" and current_metric < best_metric)
            )
            if improved:
                best_metric = current_metric
                patience_counter = 0
                path = os.path.join(self.checkpoint_dir, self.save_name)
                torch.save(self.model.state_dict(), path)
                logger.info("Saved best model to %s (metric=%.4f)", path, best_metric)
            else:
                patience_counter += 1

            # Early stopping
            if self.early_stopping and patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                break

        logger.info("Training finished. Best %s = %.4f", self.es_metric, best_metric)
        return self.history


# ---------------------------------------------------------------------------
# Classical ML Trainer
# ---------------------------------------------------------------------------

class ClassicalMLTrainer:
    """
    Train a classical ML model (sklearn) on hand-crafted features.

    Parameters
    ----------
    cfg : ExperimentConfig
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._build_model()

    def _build_model(self):
        ml_cfg = self.cfg.model.get("classical_ml", {})
        algo = ml_cfg.get("algorithm", "random_forest")

        if algo == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=ml_cfg.get("n_estimators", 200),
                max_depth=ml_cfg.get("max_depth", 15),
                n_jobs=-1,
                random_state=self.cfg.seed,
            )
        elif algo == "svm":
            from sklearn.svm import SVC
            self.model = SVC(kernel="rbf", probability=True, random_state=self.cfg.seed)
        elif algo == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=ml_cfg.get("n_estimators", 200),
                max_depth=ml_cfg.get("max_depth", 5),
                random_state=self.cfg.seed,
            )
        elif algo == "xgboost":
            # TODO: Install xgboost and use XGBClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("XGBoost not installed; falling back to GradientBoosting.")
            self.model = GradientBoostingClassifier(
                n_estimators=ml_cfg.get("n_estimators", 200),
                random_state=self.cfg.seed,
            )
        else:
            raise ValueError(f"Unknown classical ML algorithm: {algo}")

        logger.info("Built classical ML model: %s", algo)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("Training classical ML model on %d samples...", len(y_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved classical ML model to %s", path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
        logger.info("Loaded classical ML model from %s", path)
