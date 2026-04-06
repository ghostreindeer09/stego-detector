import numpy as np
from PIL import Image
import pathlib

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)

print("EMBEDDING VERIFICATION")
print("=" * 50)

for i in range(1, 6):
    cover_path = unseen_dir / f'unseen_clean_{i}.png'
    stego_path = unseen_dir / f'unseen_stego_{i}.png'
    
    # Force both to RGB — removes alpha channel issue
    cover = np.array(
        Image.open(cover_path).convert('RGB')
    ).astype(int)
    stego = np.array(
        Image.open(stego_path).convert('RGB')
    ).astype(int)
    
    # Resize stego to match cover if different sizes
    if cover.shape != stego.shape:
        stego_img = Image.open(stego_path).convert('RGB')
        stego_img = stego_img.resize(
            (cover.shape[1], cover.shape[0]),
            Image.LANCZOS
        )
        stego = np.array(stego_img).astype(int)
    
    diff     = np.sum(cover != stego)
    max_diff = np.max(np.abs(cover - stego))
    bpp      = diff / cover.size
    
    if diff == 0:
        status = "BROKEN — images identical"
    elif max_diff > 1:
        status = f"NOT LSB — max diff is {max_diff}"
    else:
        status = f"WORKING — {diff} pixels changed bpp={bpp:.4f}"
    
    print(f"Pair {i}: {status}")
    print(f"  Cover shape: {cover.shape}")
    print(f"  Stego shape: {stego.shape}")
    print(f"  Pixels different: {diff}")
    print(f"  Max difference:   {max_diff}")
    print()

print("=" * 50)
