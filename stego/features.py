import io
from typing import Dict

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import cv2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HighPassFilter(nn.Module):
    """
    Simple 3x3 Laplacian-like HPF applied per channel to emphasize noise residuals.
    """

    def __init__(self):
        super().__init__()
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        kernel = kernel.reshape(1, 1, 3, 3)
        kernel = np.repeat(kernel, 3, axis=0)  # one filter per input channel

        self.weight = nn.Parameter(torch.from_numpy(kernel), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        return F.conv2d(x, self.weight, bias=None, padding=1, groups=3)


def get_kv_kernel_5x5() -> np.ndarray:
    """
    KV-style 5x5 high-pass kernel (Kodovsky/Vojtěch family) used in steganalysis
    to extract noise residuals. Applied per channel as the first non-trainable layer.
    """
    # 5x5 second-order derivative / edge-emphasis kernel (common in SRM/KV literature)
    kv = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [-1, 2, 4, 2, -1],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    )
    # Normalize so response is in a reasonable range (optional; many papers use raw)
    return kv


class KVHighPassFilter(nn.Module):
    """
    Fixed 5x5 KV-style high-pass filter as the first non-trainable layer.
    Input: [B, 3, H, W] RGB. Output: [B, 3, H, W] residual map.
    """

    def __init__(self):
        super().__init__()
        kernel = get_kv_kernel_5x5()
        kernel = kernel.reshape(1, 1, 5, 5)
        kernel = np.repeat(kernel, 3, axis=0)  # one filter per input channel
        self.weight = nn.Parameter(torch.from_numpy(kernel), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, padding=2, groups=3)


def get_srm_kernels() -> torch.Tensor:
    """
    Very small subset of SRM high-pass filters (3 filters, 5x5).
    Real SRM uses many more; this is for prototype/demo.
    """
    k1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    k2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0],
            [0, -1, 2, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    k3 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    kernels = np.stack([k1, k2, k3], axis=0)  # [3, 5, 5]
    kernels = kernels[:, None, :, :]  # [3, 1, 5, 5]
    return torch.from_numpy(kernels)


class SRMFilterBank(nn.Module):
    """
    Small SRM-style filter bank applied to grayscale residuals.
    """

    def __init__(self):
        super().__init__()
        kernels = get_srm_kernels()
        self.weight = nn.Parameter(kernels, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] (grayscale residual)
        # output: [B, 3, H, W]
        return F.conv2d(x, self.weight, padding=2)


def dct_2d(block: np.ndarray) -> np.ndarray:
    """2D DCT on 8x8 block using OpenCV."""
    return cv2.dct(block.astype(np.float32))


def extract_dct_map(img: Image.Image, max_freq: int = 4) -> np.ndarray:
    """
    Approximate DCT-based feature map for JPEG images.
    - Converts to Y channel (luminance).
    - Splits into 8x8 blocks.
    - Computes DCT per block, keeps low-frequency coefficients up to max_freq.
    - Reassembles them into small feature maps.
    Returns: [C, H', W'] numpy array (C ~ max_freq^2).
    """
    ycbcr = img.convert("YCbCr")
    y, _, _ = ycbcr.split()
    y_np = np.array(y, dtype=np.float32)

    h, w = y_np.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    y_np = np.pad(y_np, ((0, h_pad), (0, w_pad)), mode="reflect")
    h_p, w_p = y_np.shape

    blocks_y = h_p // 8
    blocks_x = w_p // 8

    coeffs = []
    for u in range(max_freq):
        for v in range(max_freq):
            coeffs.append((u, v))
    C = len(coeffs)

    dct_maps = np.zeros((C, blocks_y, blocks_x), dtype=np.float32)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = y_np[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8]
            dct_block = dct_2d(block)
            for ci, (u, v) in enumerate(coeffs):
                dct_maps[ci, by, bx] = dct_block[u, v]

    dct_maps = (dct_maps - dct_maps.mean()) / (dct_maps.std() + 1e-6)
    return dct_maps


def ela_image(img: Image.Image, quality: int = 90) -> Image.Image:
    """
    Error Level Analysis: re-save at given JPEG quality, subtract, enhance.
    """
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    ela = ImageChops.difference(img.convert("RGB"), resaved.convert("RGB"))
    extrema = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    scale = 255.0 / max(1, max_diff)
    ela_enhanced = ImageEnhance.Brightness(ela).enhance(scale)
    return ela_enhanced


class FeatureExtractor:
    """
    Orchestrates:
    - HPF residuals
    - Spatial SRM-based residuals for PNG/BMP-like images
    - DCT-based frequency maps for JPEG
    - ELA map (for visualization / heuristic)
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size

        self.base_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

        self.hpf = HighPassFilter()
        self.srm_bank = SRMFilterBank()

        self.norm = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        x = self.base_transform(img)
        return self.norm(x)

    def compute_hpf_residual(self, img: Image.Image) -> torch.Tensor:
        x = self._to_tensor(img).unsqueeze(0)  # [1, 3, H, W]
        with torch.no_grad():
            hpf_res = self.hpf(x)
        return hpf_res

    def compute_srm_residuals(self, img: Image.Image) -> torch.Tensor:
        """
        Spatial path for non-JPEG images.
        Converts residual to grayscale then applies SRM bank.
        """
        hpf_res = self.compute_hpf_residual(img)
        gray = (
            0.2989 * hpf_res[:, 0:1]
            + 0.5870 * hpf_res[:, 1:2]
            + 0.1140 * hpf_res[:, 2:3]
        )
        with torch.no_grad():
            srm_res = self.srm_bank(gray)
        return srm_res

    def compute_dct_features(self, img: Image.Image) -> torch.Tensor:
        """
        Frequency path for JPEG images.
        Returns: [1, C, H', W'] tensor.
        """
        dct_maps = extract_dct_map(img)
        dct_tensor = torch.from_numpy(dct_maps).unsqueeze(0)
        dct_tensor = F.interpolate(
            dct_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return dct_tensor

    def compute_ela(self, img: Image.Image) -> Image.Image:
        return ela_image(img)

    def extract_features(self, img: Image.Image, image_format: str) -> Dict[str, torch.Tensor]:
        """
        Returns dict of feature maps depending on format.
        image_format: e.g., 'JPEG', 'PNG', 'BMP', etc.
        """
        image_format = (image_format or "").upper()
        is_jpeg = "JPEG" in image_format or "JPG" in image_format

        features: Dict[str, torch.Tensor] = {}
        features["hpf"] = self.compute_hpf_residual(img)

        if is_jpeg:
            features["dct"] = self.compute_dct_features(img)
        else:
            features["srm"] = self.compute_srm_residuals(img)

        return features

