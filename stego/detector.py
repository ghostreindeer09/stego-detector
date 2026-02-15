from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .features import FeatureExtractor, ela_image, get_device
from .model import SRNetBackbone, GradCAM


class StegoDetector:
    """
    High-level interface that:
    - Extracts residual features (spatial and frequency)
    - Runs SRNet-like backbone
    - Produces a threat score and Grad-CAM heatmap
    """

    def __init__(
        self,
        image_size: int = 256,
        model_weights: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.feature_extractor = FeatureExtractor(image_size=image_size)

        # HPF: 3 channels; SRM: 3 channels; DCT: 16 channels (4x4 low-freq)
        self.num_channels_spatial = 3 + 3
        self.num_channels_freq = 3 + 16
        self.in_channels = max(self.num_channels_spatial, self.num_channels_freq)

        self.model = SRNetBackbone(in_channels=self.in_channels, num_classes=1).to(
            self.device
        )

        if model_weights:
            state = torch.load(model_weights, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

        self.gradcam = GradCAM(self.model, target_layer=self.model.layer5)

    def _prepare_input_tensor(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        hpf = features["hpf"]  # [1, 3, H, W]

        if "srm" in features:
            srm = features["srm"]
            x = torch.cat([hpf, srm], dim=1)
        elif "dct" in features:
            dct = features["dct"]
            if hpf.shape[2:] != dct.shape[2:]:
                hpf = F.interpolate(
                    hpf,
                    size=dct.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([hpf, dct], dim=1)
        else:
            x = hpf

        if x.shape[1] < self.in_channels:
            pad_channels = self.in_channels - x.shape[1]
            pad = torch.zeros(
                (x.shape[0], pad_channels, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.in_channels:
            x = x[:, : self.in_channels]

        return x

    def predict(self, img: Image.Image) -> Tuple[float, torch.Tensor, Image.Image]:
        """
        Returns:
        - threat_score: float 0-100 (%)
        - heatmap: torch.Tensor [H, W] in [0,1]
        - ela_enhanced: PIL.Image
        """
        img_format = (img.format or "").upper()
        features = self.feature_extractor.extract_features(img, img_format)

        x = self._prepare_input_tensor(features).to(self.device)
        x.requires_grad_(True)

        self.model.eval()
        logits, _ = self.model(x)
        prob = torch.sigmoid(logits)[0, 0].item()
        threat_score = float(prob * 100.0)

        heatmap = self.gradcam.generate(logits)
        ela_enhanced = ela_image(img)

        return threat_score, heatmap.detach().cpu(), ela_enhanced

