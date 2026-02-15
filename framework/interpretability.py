"""
Interpretability Module
========================
Optional module providing:
    - Grad-CAM heatmap generation
    - SHAP-based feature importance (for classical ML models)

Useful for understanding what the model focuses on when detecting steganography.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM (wraps and extends existing stego.model.GradCAM)
# ---------------------------------------------------------------------------

class GradCAMAnalyzer:
    """
    Generate Grad-CAM heatmaps for a batch of images and save visualisations.

    Parameters
    ----------
    model : nn.Module (SRNet or SRNetBackbone)
    target_layer : nn.Module  (e.g. model.backbone.layer5)
    device : torch.device
    output_dir : str
    """

    def __init__(self, model, target_layer, device, output_dir: str = "interpretability/grad_cam"):
        import torch
        import torch.nn.functional as F
        from stego.model import GradCAM

        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.gradcam = GradCAM(model, target_layer)

    def generate_heatmap(self, image_tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image tensor [1, C, H, W].

        Returns
        -------
        np.ndarray of shape [H, W] in [0, 1].
        """
        import torch

        self.model.eval()
        image_tensor = image_tensor.to(self.device).requires_grad_(True)
        logits, _ = self.model(image_tensor)
        heatmap = self.gradcam.generate(logits)
        return heatmap.detach().cpu().numpy()

    def analyze_batch(
        self,
        images: List[Image.Image],
        transform,
        labels: Optional[List[int]] = None,
        prefix: str = "sample",
    ) -> List[np.ndarray]:
        """
        Generate and save Grad-CAM overlays for a list of images.

        Parameters
        ----------
        images : list of PIL images
        transform : callable that converts PIL -> tensor
        labels : optional list of true labels
        prefix : filename prefix

        Returns
        -------
        list of heatmap arrays
        """
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        heatmaps = []
        for i, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            transformed = transform(image=img_np)
            tensor = transformed["image"].unsqueeze(0)

            hmap = self.generate_heatmap(tensor)
            heatmaps.append(hmap)

            # Overlay visualisation
            hmap_resized = np.array(
                Image.fromarray((hmap * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
            ) / 255.0
            colormap = cm.get_cmap("jet")
            heatmap_color = colormap(hmap_resized)[..., :3]
            overlay = 0.5 * heatmap_color + 0.5 * (np.array(img.convert("RGB")) / 255.0)
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(hmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            label_str = ""
            if labels is not None:
                label_str = f" (Label: {'Stego' if labels[i] == 1 else 'Cover'})"
            axes[2].set_title(f"Overlay{label_str}")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"{prefix}_{i:04d}_gradcam.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Grad-CAM saved: %s", save_path)

        return heatmaps


# ---------------------------------------------------------------------------
# SHAP (for classical ML)
# ---------------------------------------------------------------------------

class SHAPAnalyzer:
    """
    SHAP-based interpretability for classical ML models.

    Parameters
    ----------
    model : sklearn model with predict_proba
    output_dir : str
    """

    def __init__(self, model, output_dir: str = "interpretability/shap"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Compute SHAP values and generate summary plot.

        Parameters
        ----------
        X_background : np.ndarray  [N_bg, D]
        X_test : np.ndarray  [N_test, D]
        feature_names : list of str
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP library not installed. Skipping SHAP analysis.")
            logger.info("Install with: pip install shap")
            return

        import matplotlib
        matplotlib.use("Agg")

        explainer = shap.Explainer(self.model, X_background, feature_names=feature_names)
        shap_values = explainer(X_test)

        # Summary plot
        save_path = os.path.join(self.output_dir, "shap_summary.png")
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", save_path)

        # Bar plot
        bar_path = os.path.join(self.output_dir, "shap_bar.png")
        shap.plots.bar(shap_values, show=False)
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar plot saved to %s", bar_path)
