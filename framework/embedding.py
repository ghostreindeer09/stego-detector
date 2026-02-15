"""
Embedding Simulator
====================
Provides steganographic embedding methods for automated dataset generation.
Currently supports LSB (Least Significant Bit) embedding.

Designed for extensibility — add new embedding methods by subclassing BaseEmbedder.
"""

import io
import logging
import random
import string
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
# LSB Embedding (pure-Python, no external library needed)
# ---------------------------------------------------------------------------

class LSBEmbedder(BaseEmbedder):
    """
    Least Significant Bit embedding on RGB channels.
    Embeds message bits sequentially across pixels (R, G, B order).
    A 32-bit binary header stores the message length for extraction.
    """

    def embed(self, image: Image.Image, message: str) -> Image.Image:
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.uint8).copy()
        flat = pixels.flatten()

        # Convert message to binary (UTF-8 bytes -> bits)
        msg_bytes = message.encode("utf-8")
        # 32-bit length header + message bits
        header = format(len(msg_bytes), "032b")
        bits = header + "".join(format(b, "08b") for b in msg_bytes)

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

        # Read 32-bit header
        header_bits = "".join(str(flat[i] & 1) for i in range(32))
        msg_len = int(header_bits, 2)

        total_bits = 32 + msg_len * 8
        if total_bits > len(flat):
            raise ValueError("Cannot extract — image too small for declared message length.")

        msg_bits = "".join(str(flat[i] & 1) for i in range(32, total_bits))
        msg_bytes = bytes(int(msg_bits[i:i+8], 2) for i in range(0, len(msg_bits), 8))
        return msg_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_EMBEDDERS = {
    "lsb": LSBEmbedder,
}


def get_embedder(method: str = "lsb") -> BaseEmbedder:
    """Return an embedder instance by name."""
    method = method.lower()
    if method not in _EMBEDDERS:
        raise ValueError(f"Unknown embedding method '{method}'. Available: {list(_EMBEDDERS.keys())}")
    return _EMBEDDERS[method]()
