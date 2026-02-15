"""
Robustness Testing Module
===========================
Tests model resilience to real-world perturbations:
    - JPEG compression at various quality levels
    - Gaussian noise at various sigma levels
    - Image resizing (up/down)
    - Cropping at various ratios
    - Varying steganographic payload sizes

Produces per-perturbation metrics for tables and plots.
"""

import io
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Re-save image with given JPEG quality and return decompressed result."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise with given standard deviation."""
    arr = np.array(img, dtype=np.float64)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_resize(img: Image.Image, scale: float) -> Image.Image:
    """Resize image by scale factor, then resize back to original dims."""
    w, h = img.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    return resized.resize((w, h), Image.BILINEAR)


def apply_crop(img: Image.Image, ratio: float) -> Image.Image:
    """Center-crop by ratio, then pad back to original size."""
    w, h = img.size
    crop_w, crop_h = int(w * ratio), int(h * ratio)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    # Pad back to original size (centre, black padding)
    result = Image.new("RGB", (w, h), (0, 0, 0))
    paste_x = (w - crop_w) // 2
    paste_y = (h - crop_h) // 2
    result.paste(cropped, (paste_x, paste_y))
    return result


# ---------------------------------------------------------------------------
# Robustness Tester
# ---------------------------------------------------------------------------

class RobustnessTester:
    """
    Runs a trained model against perturbed versions of test images
    and records metrics for each perturbation configuration.

    Parameters
    ----------
    cfg : ExperimentConfig
    model_predict_fn : callable
        Function (PIL.Image) -> float  returning probability of stego.
    """

    def __init__(self, cfg, model_predict_fn: Callable[[Image.Image], float]):
        self.cfg = cfg
        self.predict_fn = model_predict_fn
        self.rob_cfg = cfg.robustness
        self.output_dir = self.rob_cfg.get("output_dir", "robustness_results")
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_perturbation(
        self,
        images: List[Image.Image],
        labels: np.ndarray,
        perturbation_fn: Callable,
        param_name: str,
        param_values: list,
        threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """Run a perturbation sweep and collect metrics."""
        from .evaluator import compute_all_metrics

        results = {}
        for val in param_values:
            scores = []
            for img in images:
                perturbed = perturbation_fn(img, val)
                prob = self.predict_fn(perturbed)
                scores.append(prob)

            y_scores = np.array(scores)
            metrics = compute_all_metrics(labels, y_scores, threshold=threshold)
            metrics[param_name] = val
            results[f"{param_name}={val}"] = metrics

            logger.info(
                "%s=%s -> Acc=%.4f  F1=%.4f  AUC=%.4f  FP=%d  FN=%d",
                param_name, val,
                metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
                metrics["fp"], metrics["fn"],
            )

        return results

    def run_all(
        self,
        images: List[Image.Image],
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Run all enabled perturbation tests.

        Parameters
        ----------
        images : list of PIL images (test set)
        labels : np.ndarray of int labels (0=cover, 1=stego)
        threshold : float

        Returns
        -------
        dict mapping perturbation_type -> {param_val -> metrics}
        """
        perturbs = self.rob_cfg.get("perturbations", {})
        all_results = {}

        # --- JPEG Compression ---
        jpeg_cfg = perturbs.get("jpeg_compression", {})
        if jpeg_cfg.get("enabled", False):
            logger.info("=== Robustness: JPEG Compression ===")
            all_results["jpeg_compression"] = self._run_perturbation(
                images, labels,
                apply_jpeg_compression,
                "quality", jpeg_cfg.get("qualities", [70, 80, 90]),
                threshold,
            )

        # --- Gaussian Noise ---
        noise_cfg = perturbs.get("gaussian_noise", {})
        if noise_cfg.get("enabled", False):
            logger.info("=== Robustness: Gaussian Noise ===")
            all_results["gaussian_noise"] = self._run_perturbation(
                images, labels,
                apply_gaussian_noise,
                "sigma", noise_cfg.get("sigmas", [1.0, 5.0, 10.0]),
                threshold,
            )

        # --- Resize ---
        resize_cfg = perturbs.get("resize", {})
        if resize_cfg.get("enabled", False):
            logger.info("=== Robustness: Resize ===")
            all_results["resize"] = self._run_perturbation(
                images, labels,
                apply_resize,
                "scale", resize_cfg.get("scales", [0.5, 0.75, 1.5]),
                threshold,
            )

        # --- Crop ---
        crop_cfg = perturbs.get("crop", {})
        if crop_cfg.get("enabled", False):
            logger.info("=== Robustness: Crop ===")
            all_results["crop"] = self._run_perturbation(
                images, labels,
                apply_crop,
                "ratio", crop_cfg.get("ratios", [0.5, 0.7, 0.9]),
                threshold,
            )

        # --- Payload Size ---
        # NOTE: payload size robustness requires re-embedding with different sizes.
        # This is a skeleton — fill in with actual re-embedding logic.
        payload_cfg = perturbs.get("payload_size", {})
        if payload_cfg.get("enabled", False):
            logger.info("=== Robustness: Payload Size ===")
            logger.info(
                "Payload size perturbation requires re-embedding. "
                "Use dataset_generator with variable_payload=true for this test."
            )
            # TODO: Integrate re-embedding loop here for full automation
            # payload_lengths = payload_cfg.get("lengths", [4, 16, 64])
            # For each length: re-embed -> predict -> compute metrics

        # Save results
        import json
        results_path = os.path.join(self.output_dir, "robustness_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Robustness results saved to %s", results_path)

        return all_results
