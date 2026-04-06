"""
pipeline/data_gen.py — Leakage-free data generation (Part 2)

CRITICAL: Split source images FIRST, then generate stego per-split.
This prevents any image from appearing in multiple splits.
"""
import os
import hashlib
import pathlib
import random
import shutil
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from framework.embedding import (
    embed_lsb, generate_random_message, verify_lsb_constraint,
    get_available_algorithms
)

import logging
logger = logging.getLogger(__name__)

SPLIT_SEED = 42
EMBED_SEED = 123


def split_source_images(
    source_dir: str,
    train_ratio: float = 0.80,
    seed: int = SPLIT_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split source cover images into train/val/test BEFORE any stego generation.
    This is the ONLY correct way to prevent data leakage.
    """
    all_images = sorted(
        str(p) for p in pathlib.Path(source_dir).glob('*.png')
    )
    if not all_images:
        # Try jpg too
        all_images = sorted(
            str(p) for p in pathlib.Path(source_dir).glob('*.*')
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )
    total = len(all_images)
    logger.info(f"Total source images: {total}")

    train_imgs, remaining = train_test_split(
        all_images, train_size=train_ratio, random_state=seed
    )
    val_imgs, test_imgs = train_test_split(
        remaining, train_size=0.5, random_state=seed
    )

    # VERIFY zero overlap
    train_set = set(train_imgs)
    val_set = set(val_imgs)
    test_set = set(test_imgs)
    assert len(train_set & val_set) == 0, "LEAK: train ∩ val"
    assert len(train_set & test_set) == 0, "LEAK: train ∩ test"
    assert len(val_set & test_set) == 0, "LEAK: val ∩ test"
    assert len(train_set) + len(val_set) + len(test_set) == total

    logger.info(f"Split: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs


def generate_stego_for_split(
    image_list: List[str],
    split_name: str,
    output_dir: str,
    target_resolution: Tuple[int, int] = (256, 256),
    algorithms: Optional[List[str]] = None,
    deterministic: bool = True,
    payload_range: Tuple[int, int] = (2000, 10000),
    embed_seed: int = EMBED_SEED,
) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    Generate cover/stego pairs for a single split.
    Returns list of (cover_path, stego_path) pairs and bpp values.
    """
    if algorithms is None:
        algorithms = ['lsb_sequential']

    cover_out = os.path.join(output_dir, split_name, 'cover')
    stego_out = os.path.join(output_dir, split_name, 'stego')
    os.makedirs(cover_out, exist_ok=True)
    os.makedirs(stego_out, exist_ok=True)

    pairs = []
    bpp_values = []
    success = 0
    fail = 0
    algo_counts = {a: 0 for a in algorithms}
    rng = random.Random(embed_seed)

    for idx, img_path in enumerate(image_list):
        img_name = pathlib.Path(img_path).stem + '.png'
        cover_save = os.path.join(cover_out, img_name)
        stego_save = os.path.join(stego_out, img_name)

        # Skip if already generated
        if os.path.exists(cover_save) and os.path.exists(stego_save):
            pairs.append((cover_save, stego_save))
            # Quick bpp check on existing pair
            try:
                c = np.array(Image.open(cover_save))
                s = np.array(Image.open(stego_save))
                bpp_values.append(np.sum(c != s) / c.size)
            except Exception:
                pass
            success += 1
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)

            # Resize/crop to target resolution if needed
            h, w = img_arr.shape[:2]
            tw, th = target_resolution
            if w != tw or h != th:
                if w >= tw and h >= th:
                    # Center crop
                    left = (w - tw) // 2
                    top = (h - th) // 2
                    img_arr = img_arr[top:top+th, left:left+tw]
                else:
                    img_arr = np.array(img.resize(target_resolution))

            # Save cover
            Image.fromarray(img_arr).save(cover_save, format='PNG')

            # Select algorithm
            if deterministic:
                algo = algorithms[0]
            else:
                algo = algorithms[idx % len(algorithms)]

            # Generate message
            plen = rng.randint(payload_range[0], payload_range[1])
            img_seed = embed_seed + idx
            message = generate_random_message(length=plen, seed=img_seed)

            # Embed
            stego_arr = embed_lsb(img_arr, message, algorithm=algo, seed=img_seed)

            # Verify LSB constraint
            result = verify_lsb_constraint(img_arr, stego_arr)
            if not result['passed']:
                logger.warning(f"FAIL {img_name}: max_diff={result['max_diff']} "
                               f"changed={result['changed_pixels']}")
                fail += 1
                continue

            Image.fromarray(stego_arr).save(stego_save, format='PNG')
            pairs.append((cover_save, stego_save))
            bpp_values.append(result['bpp'])
            algo_counts[algo] += 1
            success += 1

            if success % 200 == 0:
                logger.info(f"  [{split_name}] Generated {success}/{len(image_list)}...")

        except Exception as e:
            logger.warning(f"FAIL {img_path}: {e}")
            fail += 1

    rate = success / max(1, len(image_list)) * 100
    logger.info(f"  {split_name}: {success}/{len(image_list)} ({rate:.1f}%) "
                f"| bpp={np.mean(bpp_values):.4f} | algos={algo_counts}")
    return pairs, bpp_values


def run_leakage_detection(data_dir: str, splits=('train', 'val', 'test')):
    """Verify zero data leakage across all splits."""
    logger.info("RUNNING LEAKAGE DETECTION SUITE...")
    split_hashes = {}

    for split in splits:
        cover_dir = os.path.join(data_dir, split, 'cover')
        if not os.path.isdir(cover_dir):
            continue
        paths = sorted(pathlib.Path(cover_dir).glob('*.png'))
        names = set(p.name for p in paths)
        # Also hash file contents for a sample
        content_hashes = set()
        for p in paths[:200]:
            content_hashes.add(hashlib.md5(open(p, 'rb').read()).hexdigest())
        split_hashes[split] = {'names': names, 'hashes': content_hashes, 'count': len(paths)}
        logger.info(f"  {split}: {len(paths)} images")

    # Check name overlap
    for a in splits:
        for b in splits:
            if a >= b or a not in split_hashes or b not in split_hashes:
                continue
            name_overlap = split_hashes[a]['names'] & split_hashes[b]['names']
            if name_overlap:
                raise RuntimeError(f"LEAK: {len(name_overlap)} shared names between {a}/{b}")
            hash_overlap = split_hashes[a]['hashes'] & split_hashes[b]['hashes']
            if hash_overlap:
                raise RuntimeError(f"LEAK: {len(hash_overlap)} content matches between {a}/{b}")

    logger.info("  ALL LEAKAGE CHECKS: PASS ✓")
    return True


def build_leakage_free_dataset(
    source_dir: str,
    output_dir: str,
    target_resolution: Tuple[int, int] = (256, 256),
    train_algorithms: Optional[List[str]] = None,
    payload_range: Tuple[int, int] = (2000, 10000),
    split_seed: int = SPLIT_SEED,
    embed_seed: int = EMBED_SEED,
) -> dict:
    """
    Complete leakage-free dataset generation:
    1. Split source images
    2. Generate stego per-split
    3. Verify zero leakage
    """
    if train_algorithms is None:
        train_algorithms = get_available_algorithms()

    logger.info("=" * 62)
    logger.info("STEP 2: LEAKAGE-FREE DATA PIPELINE")
    logger.info("=" * 62)

    # Step 2.1: Split
    train_imgs, val_imgs, test_imgs = split_source_images(
        source_dir, seed=split_seed
    )

    # Step 2.2: Generate stego per-split
    # Val/test: deterministic, single algorithm
    logger.info("\nGenerating val stego (deterministic)...")
    val_pairs, val_bpp = generate_stego_for_split(
        val_imgs, 'val', output_dir, target_resolution,
        algorithms=['lsb_sequential'], deterministic=True,
        payload_range=payload_range, embed_seed=embed_seed
    )

    logger.info("\nGenerating test stego (deterministic)...")
    test_pairs, test_bpp = generate_stego_for_split(
        test_imgs, 'test', output_dir, target_resolution,
        algorithms=['lsb_sequential'], deterministic=True,
        payload_range=payload_range, embed_seed=embed_seed
    )

    # Train: multi-algorithm
    logger.info(f"\nGenerating train stego (algorithms: {train_algorithms})...")
    train_pairs, train_bpp = generate_stego_for_split(
        train_imgs, 'train', output_dir, target_resolution,
        algorithms=train_algorithms, deterministic=False,
        payload_range=payload_range, embed_seed=embed_seed
    )

    # Step 2.3: Leakage detection
    logger.info("\n")
    run_leakage_detection(output_dir)

    return {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'test_pairs': test_pairs,
        'train_bpp': train_bpp,
        'val_bpp': val_bpp,
        'test_bpp': test_bpp,
    }
