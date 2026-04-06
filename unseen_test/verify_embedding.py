# verify_embedding.py
import numpy as np
from PIL import Image
import pathlib

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)

print("EMBEDDING VERIFICATION")
print("=" * 50)

for i in range(1, 6):
    cover_path = unseen_dir / f'clean_{i}.png'
    stego_path = unseen_dir / f'stego_{i}.png'
    
    cover = np.array(Image.open(cover_path)).astype(int)
    stego = np.array(Image.open(stego_path)).astype(int)
    
    diff     = np.sum(cover != stego)
    max_diff = np.max(np.abs(cover - stego))
    
    if diff == 0:
        status = "BROKEN — no changes"
    elif max_diff > 1:
        status = f"NOT LSB — max diff is {max_diff}"
    else:
        status = f"WORKING — {diff} pixels changed"
    
    print(f"Pair {i}: {status}")

print("=" * 50)