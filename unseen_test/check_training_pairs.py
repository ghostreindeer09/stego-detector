# check_training_pairs.py
import numpy as np
from PIL import Image
import pathlib
import random

data_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data'
)

print("TRAINING DATA PAIR VERIFICATION")
print("=" * 50)

# Check train split pairs
cover_dir = data_dir / 'train' / 'cover'
stego_dir = data_dir / 'train' / 'stego'

covers = sorted(cover_dir.rglob('*.png'))
stegos = sorted(stego_dir.rglob('*.png'))

print(f"Train covers: {len(covers)}")
print(f"Train stegos: {len(stegos)}")

# Sample 10 random pairs
sample_covers = random.sample(covers, 10)

broken  = 0
working = 0
not_lsb = 0

for cover_path in sample_covers:
    stego_path = stego_dir / cover_path.name
    
    if not stego_path.exists():
        print(f"MISSING PAIR: {cover_path.name}")
        broken += 1
        continue
    
    cover = np.array(
        Image.open(cover_path).convert('RGB')
    ).astype(int)
    stego = np.array(
        Image.open(stego_path).convert('RGB')
    ).astype(int)
    
    diff     = np.sum(cover != stego)
    max_diff = np.max(np.abs(cover - stego))
    bpp      = diff / cover.size
    
    if diff == 0:
        status = "BROKEN — identical"
        broken += 1
    elif max_diff > 1:
        status = f"NOT LSB — max diff {max_diff}"
        not_lsb += 1
    else:
        status = f"OK — bpp={bpp:.4f}"
        working += 1
    
    print(f"{cover_path.name}: {status}")

print(f"\nSummary:")
print(f"  Working LSB: {working}/10")
print(f"  Broken:      {broken}/10")
print(f"  Not LSB:     {not_lsb}/10")
print("=" * 50)