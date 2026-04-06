"""
Embedding Simulator v3.1
=========================
Provides steganographic embedding methods for automated dataset generation.
Supports 4 embedding algorithms:
  - lsb_sequential: Classic sequential LSB replacement
  - lsb_random:     Pseudo-random pixel selection for LSB replacement
  - lsb_pvd:        Pixel Value Differencing-inspired LSB embedding
  - lsb_matching:   LSB matching (±1 modification)

All algorithms guarantee max pixel change of ±1 (LSB constraint).
"""

import io
import logging
import random
import string
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Available algorithms
# ---------------------------------------------------------------------------

AVAILABLE_ALGORITHMS = [
    'lsb_sequential',
    'lsb_random',
    'lsb_pvd',
    'lsb_matching',
]


def get_available_algorithms() -> List[str]:
    """Return list of all implemented embedding algorithm names."""
    return list(AVAILABLE_ALGORITHMS)


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------

def generate_random_message(
    length: int = 16,
    charset: str = "alphanumeric",
    seed: Optional[int] = None,
) -> str:
    """
    Generate a random message string.

    Parameters
    ----------
    length : int
        Number of characters.
    charset : str
        One of 'alphanumeric', 'alpha', 'digits', 'custom'.
    seed : int, optional
        For reproducibility.
    """
    rng = random.Random(seed)
    if charset == "alphanumeric":
        pool = string.ascii_letters + string.digits
    elif charset == "alpha":
        pool = string.ascii_letters
    elif charset == "digits":
        pool = string.digits
    else:
        pool = string.ascii_letters + string.digits + string.punctuation
    return "".join(rng.choices(pool, k=length))


def message_to_bits(message: str) -> str:
    """Convert a message string to binary bit string with 32-bit length header."""
    msg_bytes = message.encode("utf-8")
    header = format(len(msg_bytes), "032b")
    bits = header + "".join(format(b, "08b") for b in msg_bytes)
    return bits


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for steganographic embedding."""

    @abstractmethod
    def embed(self, image: Image.Image, message: str) -> Image.Image:
        """Embed *message* into *image* and return the stego image."""
        ...

    @abstractmethod
    def extract(self, image: Image.Image, length: int) -> str:
        """Extract a message of *length* characters from *image*."""
        ...


# ---------------------------------------------------------------------------
# LSB Sequential Embedding
# ---------------------------------------------------------------------------

class LSBSequentialEmbedder(BaseEmbedder):
    """
    Least Significant Bit embedding on RGB channels.
    Embeds message bits sequentially across pixels (R, G, B order).
    A 32-bit binary header stores the message length for extraction.
    Max pixel change: ±1 (LSB only).
    """

    def embed(self, image: Image.Image, message: str) -> Image.Image:
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8).copy()
        flat = pixels.flatten()

        bits = message_to_bits(message)

        if len(bits) > len(flat):
            raise ValueError(
                f"Message too large for image: need {len(bits)} pixels, "
                f"have {len(flat)}"
            )

        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | int(bit)

        pixels = flat.reshape(pixels.shape)
        return Image.fromarray(pixels, mode="RGB")

    def extract(self, image: Image.Image, length: int = 0) -> str:
        img = image.convert("RGB")
        flat = np.array(img, dtype=np.uint8).flatten()

        header_bits = "".join(str(flat[i] & 1) for i in range(32))
        msg_len = int(header_bits, 2)

        total_bits = 32 + msg_len * 8
        if total_bits > len(flat):
            raise ValueError("Cannot extract — image too small for declared message length.")

        msg_bits = "".join(str(flat[i] & 1) for i in range(32, total_bits))
        msg_bytes = bytes(int(msg_bits[i:i+8], 2) for i in range(0, len(msg_bits), 8))
        return msg_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# LSB Random Embedding
# ---------------------------------------------------------------------------

class LSBRandomEmbedder(BaseEmbedder):
    """
    LSB embedding with pseudo-random pixel selection.
    Uses a seed-based permutation to scatter bits across the image.
    This makes the embedding more spatially distributed vs sequential.
    Max pixel change: ±1 (LSB only).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def embed(self, image: Image.Image, message: str) -> Image.Image:
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8).copy()
        flat = pixels.flatten()

        bits = message_to_bits(message)

        if len(bits) > len(flat):
            raise ValueError(
                f"Message too large for image: need {len(bits)} subpixels, "
                f"have {len(flat)}"
            )

        # Generate pseudo-random permutation of pixel indices
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(flat))

        for bit_idx, bit in enumerate(bits):
            pixel_idx = indices[bit_idx]
            flat[pixel_idx] = (flat[pixel_idx] & 0xFE) | int(bit)

        pixels = flat.reshape(pixels.shape)
        return Image.fromarray(pixels, mode="RGB")

    def extract(self, image: Image.Image, length: int = 0) -> str:
        img = image.convert("RGB")
        flat = np.array(img, dtype=np.uint8).flatten()

        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(flat))

        # Read 32-bit header
        header_bits = "".join(str(flat[indices[i]] & 1) for i in range(32))
        msg_len = int(header_bits, 2)

        total_bits = 32 + msg_len * 8
        if total_bits > len(flat):
            raise ValueError("Cannot extract — image too small for declared message length.")

        msg_bits = "".join(str(flat[indices[i]] & 1) for i in range(32, total_bits))
        msg_bytes = bytes(int(msg_bits[j:j+8], 2) for j in range(0, len(msg_bits), 8))
        return msg_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# LSB PVD (Pixel Value Differencing-inspired) Embedding
# ---------------------------------------------------------------------------

class LSBPVDEmbedder(BaseEmbedder):
    """
    PVD-inspired LSB embedding: embeds bits preferentially in pixels
    adjacent to edges (high local gradient), where changes are harder
    to detect. Falls back to sequential for remaining bits.
    Max pixel change: ±1 (LSB only).
    """

    def embed(self, image: Image.Image, message: str) -> Image.Image:
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8).copy()
        h, w, c = pixels.shape
        flat = pixels.flatten()

        bits = message_to_bits(message)

        if len(bits) > len(flat):
            raise ValueError(
                f"Message too large for image: need {len(bits)} subpixels, "
                f"have {len(flat)}"
            )

        # Compute local gradient magnitude per pixel (on grayscale)
        gray = 0.299 * pixels[:, :, 0].astype(float) + \
               0.587 * pixels[:, :, 1].astype(float) + \
               0.114 * pixels[:, :, 2].astype(float)

        # Sobel-like gradient
        grad = np.zeros_like(gray)
        grad[1:-1, 1:-1] = (
            np.abs(gray[1:-1, 2:] - gray[1:-1, :-2]) +
            np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1])
        )

        # Create priority ordering: high-gradient pixels first
        # Flatten gradient and expand to 3 channels
        grad_flat = np.repeat(grad.flatten(), 3)  # [h*w*3]
        priority = np.argsort(-grad_flat)  # highest gradient first

        # Embed bits in priority order
        for bit_idx, bit in enumerate(bits):
            pixel_idx = priority[bit_idx]
            flat[pixel_idx] = (flat[pixel_idx] & 0xFE) | int(bit)

        pixels = flat.reshape(pixels.shape)
        return Image.fromarray(pixels, mode="RGB")

    def extract(self, image: Image.Image, length: int = 0) -> str:
        # PVD extraction requires the same gradient-based ordering
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8)
        h, w, c = pixels.shape
        flat = pixels.flatten()

        gray = 0.299 * pixels[:, :, 0].astype(float) + \
               0.587 * pixels[:, :, 1].astype(float) + \
               0.114 * pixels[:, :, 2].astype(float)

        grad = np.zeros_like(gray)
        grad[1:-1, 1:-1] = (
            np.abs(gray[1:-1, 2:] - gray[1:-1, :-2]) +
            np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1])
        )

        grad_flat = np.repeat(grad.flatten(), 3)
        priority = np.argsort(-grad_flat)

        header_bits = "".join(str(flat[priority[i]] & 1) for i in range(32))
        msg_len = int(header_bits, 2)

        total_bits = 32 + msg_len * 8
        msg_bits = "".join(str(flat[priority[i]] & 1) for i in range(32, total_bits))
        msg_bytes = bytes(int(msg_bits[j:j+8], 2) for j in range(0, len(msg_bits), 8))
        return msg_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# LSB Matching Embedding
# ---------------------------------------------------------------------------

class LSBMatchingEmbedder(BaseEmbedder):
    """
    LSB matching (±1 modification): instead of forcing the LSB to match
    the message bit, this method randomly adds or subtracts 1 when the
    LSB doesn't match. This creates a more natural noise distribution
    compared to LSB replacement.
    Max pixel change: ±1.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def embed(self, image: Image.Image, message: str) -> Image.Image:
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8).copy()
        flat = pixels.astype(np.int16).flatten()

        bits = message_to_bits(message)

        if len(bits) > len(flat):
            raise ValueError(
                f"Message too large for image: need {len(bits)} subpixels, "
                f"have {len(flat)}"
            )

        rng = np.random.RandomState(self.seed)

        for i, bit in enumerate(bits):
            bit_val = int(bit)
            if (flat[i] & 1) != bit_val:
                # LSB doesn't match — randomly add or subtract 1
                if flat[i] == 0:
                    flat[i] = 1  # can only go up
                elif flat[i] == 255:
                    flat[i] = 254  # can only go down
                else:
                    delta = rng.choice([-1, 1])
                    flat[i] += delta

        flat = np.clip(flat, 0, 255).astype(np.uint8)
        pixels = flat.reshape(pixels.shape)
        return Image.fromarray(pixels, mode="RGB")

    def extract(self, image: Image.Image, length: int = 0) -> str:
        # Extraction is identical to sequential LSB — we read the LSB
        img = image.convert("RGB")
        flat = np.array(img, dtype=np.uint8).flatten()

        header_bits = "".join(str(flat[i] & 1) for i in range(32))
        msg_len = int(header_bits, 2)

        total_bits = 32 + msg_len * 8
        if total_bits > len(flat):
            raise ValueError("Cannot extract — image too small for declared message length.")

        msg_bits = "".join(str(flat[i] & 1) for i in range(32, total_bits))
        msg_bytes = bytes(int(msg_bits[j:j+8], 2) for j in range(0, len(msg_bits), 8))
        return msg_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def verify_lsb_constraint(cover_array: np.ndarray, stego_array: np.ndarray) -> dict:
    """
    Verify that the stego image satisfies LSB constraint:
    - Max pixel difference is ±1
    - At least some pixels changed
    Returns dict with verification results.
    """
    cover = cover_array.astype(np.int16)
    stego = stego_array.astype(np.int16)
    diff = cover - stego

    max_diff = int(np.max(np.abs(diff)))
    changed_pixels = int(np.sum(diff != 0))
    total_pixels = int(cover.size)
    bpp = changed_pixels / max(1, total_pixels)

    return {
        'max_diff': max_diff,
        'changed_pixels': changed_pixels,
        'total_pixels': total_pixels,
        'bpp': bpp,
        'lsb_valid': max_diff <= 1,
        'has_changes': changed_pixels > 0,
        'passed': max_diff <= 1 and changed_pixels > 0,
    }


# ---------------------------------------------------------------------------
# Unified embedding function
# ---------------------------------------------------------------------------

def embed_lsb(
    cover_array: np.ndarray,
    message: str,
    algorithm: str = 'lsb_sequential',
    seed: int = 42,
) -> np.ndarray:
    """
    Embed a message into a cover image array using the specified algorithm.

    Parameters
    ----------
    cover_array : np.ndarray
        Cover image as uint8 array [H, W, 3].
    message : str
        Message to embed.
    algorithm : str
        One of: lsb_sequential, lsb_random, lsb_pvd, lsb_matching
    seed : int
        Seed for algorithms that use randomness.

    Returns
    -------
    np.ndarray
        Stego image as uint8 array [H, W, 3].
    """
    cover_img = Image.fromarray(cover_array, mode="RGB")

    if algorithm == 'lsb_sequential':
        embedder = LSBSequentialEmbedder()
    elif algorithm == 'lsb_random':
        embedder = LSBRandomEmbedder(seed=seed)
    elif algorithm == 'lsb_pvd':
        embedder = LSBPVDEmbedder()
    elif algorithm == 'lsb_matching':
        embedder = LSBMatchingEmbedder(seed=seed)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {AVAILABLE_ALGORITHMS}"
        )

    stego_img = embedder.embed(cover_img, message)
    return np.array(stego_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Factory (backwards compat)
# ---------------------------------------------------------------------------

# Keep LSBEmbedder as alias for backwards compatibility
LSBEmbedder = LSBSequentialEmbedder

_EMBEDDERS = {
    "lsb": LSBSequentialEmbedder,
    "lsb_sequential": LSBSequentialEmbedder,
    "lsb_random": LSBRandomEmbedder,
    "lsb_pvd": LSBPVDEmbedder,
    "lsb_matching": LSBMatchingEmbedder,
}


def get_embedder(method: str = "lsb") -> BaseEmbedder:
    """Return an embedder instance by name."""
    method = method.lower()
    if method not in _EMBEDDERS:
        raise ValueError(f"Unknown embedding method '{method}'. Available: {list(_EMBEDDERS.keys())}")
    return _EMBEDDERS[method]()
