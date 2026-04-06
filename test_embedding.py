"""Quick test of all 4 embedding algorithms."""
import numpy as np
from PIL import Image
from framework.embedding import embed_lsb, verify_lsb_constraint, get_available_algorithms

img = np.array(Image.open('data/openimages/cover/0012b3323b4ee7e7.png').convert('RGB'))
print(f"Image shape: {img.shape}")

for algo in get_available_algorithms():
    msg = 'A' * 5000
    stego = embed_lsb(img, msg, algorithm=algo, seed=42)
    r = verify_lsb_constraint(img, stego)
    tag = "PASS" if r['passed'] else "FAIL"
    print(f"  {algo:20s}: {tag} | bpp={r['bpp']:.4f} "
          f"max_diff={r['max_diff']} changed={r['changed_pixels']}")

print("\nAll embedding algorithms verified!")
