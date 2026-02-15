"""
Automated Stego Dataset Generator
===================================
Takes a folder of cover images, generates random messages, embeds them
using a configurable steganographic embedding method, and produces:

    1. stego_images/   — folder of stego images
    2. dataset_mapping.csv — CSV with (image_name, label, payload)
       label=0 for cover, label=1 for stego.

Fully automated, scalable, reproducible, and ready for ML training.
"""

import csv
import logging
import os
import random
from typing import Optional

from PIL import Image

from .embedding import generate_random_message, get_embedder

logger = logging.getLogger(__name__)

# Supported input extensions
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def generate_stego_dataset(
    cover_dir: str,
    stego_dir: str,
    mapping_csv: str,
    embedding_method: str = "lsb",
    default_payload_length: int = 16,
    payload_charset: str = "alphanumeric",
    variable_payload: bool = False,
    payload_range: tuple = (8, 64),
    image_format: str = "png",
    seed: int = 42,
) -> int:
    """
    Generate a stego dataset from a folder of cover images.

    Parameters
    ----------
    cover_dir : str
        Directory containing cover images.
    stego_dir : str
        Directory to write stego images to.
    mapping_csv : str
        Path to the output CSV mapping file.
    embedding_method : str
        Embedding algorithm name (default: 'lsb').
    default_payload_length : int
        Default number of characters in the embedded message.
    payload_charset : str
        Character set for random messages.
    variable_payload : bool
        If True, randomly vary payload length per image within *payload_range*.
    payload_range : tuple of (int, int)
        (min_length, max_length) when variable_payload is True.
    image_format : str
        Output format for stego images (png recommended to avoid lossy recompression).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    int
        Number of stego images generated.
    """
    os.makedirs(stego_dir, exist_ok=True)
    os.makedirs(os.path.dirname(mapping_csv) or ".", exist_ok=True)

    embedder = get_embedder(embedding_method)
    rng = random.Random(seed)

    # Collect cover images
    cover_files = sorted(
        f for f in os.listdir(cover_dir)
        if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS
    )

    if not cover_files:
        raise FileNotFoundError(f"No images found in {cover_dir}")

    logger.info(
        "Found %d cover images in '%s'. Generating stego dataset...",
        len(cover_files), cover_dir,
    )

    rows = []
    generated = 0

    for idx, filename in enumerate(cover_files):
        cover_path = os.path.join(cover_dir, filename)
        name_no_ext = os.path.splitext(filename)[0]

        # --- Cover entry ---
        rows.append({
            "image_name": filename,
            "label": 0,
            "payload": "",
            "payload_length": 0,
            "source_path": cover_path,
        })

        # --- Stego entry ---
        if variable_payload:
            payload_len = rng.randint(payload_range[0], payload_range[1])
        else:
            payload_len = default_payload_length

        # Deterministic seed per image for reproducibility
        msg_seed = seed + idx
        message = generate_random_message(
            length=payload_len,
            charset=payload_charset,
            seed=msg_seed,
        )

        try:
            cover_img = Image.open(cover_path).convert("RGB")
            stego_img = embedder.embed(cover_img, message)

            stego_filename = f"{name_no_ext}_stego.{image_format}"
            stego_path = os.path.join(stego_dir, stego_filename)
            stego_img.save(stego_path, format=image_format.upper())

            rows.append({
                "image_name": stego_filename,
                "label": 1,
                "payload": message,
                "payload_length": payload_len,
                "source_path": stego_path,
            })
            generated += 1

        except Exception as e:
            logger.warning("Failed to embed in '%s': %s", filename, e)

    # Write CSV
    fieldnames = ["image_name", "label", "payload", "payload_length", "source_path"]
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Dataset generation complete: %d stego images, %d total entries in '%s'.",
        generated, len(rows), mapping_csv,
    )
    return generated


def generate_from_config(cfg) -> int:
    """
    Generate a stego dataset using values from an ExperimentConfig.

    Parameters
    ----------
    cfg : ExperimentConfig

    Returns
    -------
    int
        Number of stego images generated.
    """
    ds = cfg.dataset
    return generate_stego_dataset(
        cover_dir=ds.get("cover_images_dir", "cover_images"),
        stego_dir=ds.get("stego_images_dir", "stego_images"),
        mapping_csv=ds.get("mapping_csv", "dataset_mapping.csv"),
        embedding_method=ds.get("embedding_method", "lsb"),
        default_payload_length=ds.get("default_payload_length", 16),
        payload_charset=ds.get("payload_charset", "alphanumeric"),
        variable_payload=ds.get("variable_payload", False),
        payload_range=tuple(ds.get("payload_range", [8, 64])),
        image_format=ds.get("image_format", "png"),
        seed=ds.get("seed", 42),
    )
