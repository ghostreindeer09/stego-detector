"""
Classical ML Feature Extractor
================================
Extracts hand-crafted features for classical ML steganalysis models
(Random Forest, SVM, Gradient Boosting, XGBoost).

Feature sets:
    - histogram     : Colour channel histograms
    - cooccurrence  : Grey-level co-occurrence matrix statistics
    - dct_stats     : DCT coefficient statistics (mean, var, skew, kurtosis)
    - spatial_rich  : Spatial Rich Model (SRM) residual statistics
    - edge_density  : Canny edge density features
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _img_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float64)


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def extract_histogram_features(img_np: np.ndarray, bins: int = 64) -> np.ndarray:
    """Per-channel histograms concatenated into a single vector."""
    features = []
    for c in range(3):
        hist, _ = np.histogram(img_np[:, :, c], bins=bins, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-10
        features.append(hist)
    return np.concatenate(features)


def extract_cooccurrence_features(
    img_np: np.ndarray,
    distances: list = None,
    levels: int = 256,
) -> np.ndarray:
    """
    Grey-level co-occurrence matrix (GLCM) statistics.
    Uses simple horizontal co-occurrence for speed.
    """
    if distances is None:
        distances = [1, 2]

    gray = np.mean(img_np, axis=2).astype(np.uint8)
    features = []

    for d in distances:
        # Horizontal co-occurrence
        left = gray[:, :-d]
        right = gray[:, d:]

        # Build GLCM (downscaled to 32 levels for speed)
        scale = levels // 32
        l_q = (left // scale).astype(np.int32)
        r_q = (right // scale).astype(np.int32)
        glcm = np.zeros((32, 32), dtype=np.float64)
        for i, j in zip(l_q.ravel(), r_q.ravel()):
            glcm[i, j] += 1
        glcm /= glcm.sum() + 1e-10

        # Haralick-like features
        mean_val = glcm.mean()
        contrast = np.sum(
            glcm * (np.arange(32)[:, None] - np.arange(32)[None, :]) ** 2
        )
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum(
            glcm / (1 + np.abs(np.arange(32)[:, None] - np.arange(32)[None, :]))
        )
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

        features.extend([mean_val, contrast, energy, homogeneity, entropy])

    return np.array(features, dtype=np.float64)


def extract_dct_stats(img_np: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Statistics of DCT coefficients (mean, var, skew, kurtosis per channel)."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available; returning zeros for DCT stats.")
        return np.zeros(12, dtype=np.float64)

    features = []
    for c in range(3):
        channel = img_np[:, :, c].astype(np.float32)
        h, w = channel.shape
        h = h - h % block_size
        w = w - w % block_size
        channel = channel[:h, :w]

        coeffs = []
        for by in range(0, h, block_size):
            for bx in range(0, w, block_size):
                block = channel[by:by + block_size, bx:bx + block_size]
                dct_block = cv2.dct(block)
                coeffs.append(dct_block.ravel())

        coeffs = np.concatenate(coeffs)
        mean = np.mean(coeffs)
        var = np.var(coeffs)
        skew = float(np.mean(((coeffs - mean) / (np.sqrt(var) + 1e-10)) ** 3))
        kurt = float(np.mean(((coeffs - mean) / (np.sqrt(var) + 1e-10)) ** 4) - 3.0)
        features.extend([mean, var, skew, kurt])

    return np.array(features, dtype=np.float64)


def extract_edge_density(img_np: np.ndarray) -> np.ndarray:
    """Canny edge density features."""
    try:
        import cv2
    except ImportError:
        return np.zeros(3, dtype=np.float64)

    gray = np.mean(img_np, axis=2).astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255.0)
    mean_edge = edges.mean() / 255.0
    std_edge = edges.std() / 255.0
    return np.array([density, mean_edge, std_edge], dtype=np.float64)


def extract_spatial_rich_features(img_np: np.ndarray) -> np.ndarray:
    """
    Simple SRM-like residual statistics (mean, var, skew of high-pass filtered image).
    """
    gray = np.mean(img_np, axis=2)
    # Laplacian-like kernel
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    from scipy.signal import convolve2d
    residual = convolve2d(gray, kernel, mode="same", boundary="symm")

    mean = np.mean(residual)
    var = np.var(residual)
    skew = float(np.mean(((residual - mean) / (np.sqrt(var) + 1e-10)) ** 3))
    kurt = float(np.mean(((residual - mean) / (np.sqrt(var) + 1e-10)) ** 4) - 3.0)
    abs_mean = np.mean(np.abs(residual))

    return np.array([mean, var, skew, kurt, abs_mean], dtype=np.float64)


# ---------------------------------------------------------------------------
# Unified extractor
# ---------------------------------------------------------------------------

_FEATURE_REGISTRY = {
    "histogram": extract_histogram_features,
    "cooccurrence": extract_cooccurrence_features,
    "dct_stats": extract_dct_stats,
    "edge_density": extract_edge_density,
    "spatial_rich": extract_spatial_rich_features,
}


def extract_features(
    image: Image.Image,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract a concatenated feature vector from an image.

    Parameters
    ----------
    image : PIL.Image
    feature_names : list of str
        Which feature sets to extract. Default: all.

    Returns
    -------
    np.ndarray  — 1-D feature vector
    """
    if feature_names is None:
        feature_names = list(_FEATURE_REGISTRY.keys())

    img_np = _img_to_np(image)
    vectors = []
    for name in feature_names:
        fn = _FEATURE_REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown feature '%s'; skipping.", name)
            continue
        vectors.append(fn(img_np))

    if not vectors:
        raise ValueError("No valid features extracted.")

    return np.concatenate(vectors)


def extract_features_batch(
    images: List[Image.Image],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract features for a list of images; returns [N, D] array."""
    return np.stack([extract_features(img, feature_names) for img in images])
