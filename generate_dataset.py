#!/usr/bin/env python3
"""
generate_dataset.py
====================
CLI script to generate a stego dataset from cover images.

Usage:
    python generate_dataset.py --config configs/default_experiment.yaml
    python generate_dataset.py --cover-dir cover_images/ --stego-dir stego_images/
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.config import load_config
from framework.dataset_generator import generate_stego_dataset, generate_from_config


def main():
    parser = argparse.ArgumentParser(description="Generate stego dataset from cover images.")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path.")
    parser.add_argument("--cover-dir", type=str, default="cover_images", help="Cover images directory.")
    parser.add_argument("--stego-dir", type=str, default="stego_images", help="Output stego directory.")
    parser.add_argument("--mapping-csv", type=str, default="dataset_mapping.csv", help="Output CSV path.")
    parser.add_argument("--method", type=str, default="lsb", help="Embedding method.")
    parser.add_argument("--payload-length", type=int, default=16, help="Default message length.")
    parser.add_argument("--variable-payload", action="store_true", help="Vary payload per image.")
    parser.add_argument("--payload-min", type=int, default=8, help="Min payload (when variable).")
    parser.add_argument("--payload-max", type=int, default=64, help="Max payload (when variable).")
    parser.add_argument("--format", type=str, default="png", help="Output image format.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.config:
        cfg = load_config(args.config)
        count = generate_from_config(cfg)
    else:
        count = generate_stego_dataset(
            cover_dir=args.cover_dir,
            stego_dir=args.stego_dir,
            mapping_csv=args.mapping_csv,
            embedding_method=args.method,
            default_payload_length=args.payload_length,
            payload_charset="alphanumeric",
            variable_payload=args.variable_payload,
            payload_range=(args.payload_min, args.payload_max),
            image_format=args.format,
            seed=args.seed,
        )

    print(f"\nDone! Generated {count} stego images.")


if __name__ == "__main__":
    main()
