# reembed_unseen.py
import numpy as np
from PIL import Image
import pathlib
import random

def embed_lsb_at_target_bpp(image_path, target_bpp=0.12):
    img      = np.array(
        Image.open(image_path).convert('RGB')
    )
    h, w, c  = img.shape
    total_pixels = h * w * c
    
    # Calculate how many bits to embed
    n_bits = int(total_pixels * target_bpp)
    
    # Generate random message bits
    message_bits = np.random.randint(0, 2, n_bits)
    
    # Flatten image
    flat = img.flatten().copy()
    
    # Embed in random positions
    positions = random.sample(range(len(flat)), n_bits)
    positions.sort()
    
    for idx, pos in enumerate(positions):
        # Replace LSB with message bit
        flat[pos] = (flat[pos] & 0xFE) | message_bits[idx]
    
    stego = flat.reshape(h, w, c)
    
    # Verify
    diff = np.sum(img != stego)
    bpp  = diff / total_pixels
    
    return stego, bpp

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)

print("RE-EMBEDDING AT CORRECT PAYLOAD DENSITY")
print(f"Target bpp: 0.12 (matching training data)")
print("=" * 50)

for i in range(1, 6):
    cover_path = unseen_dir / f'unseen_clean_{i}.png'
    stego_path = unseen_dir / f'unseen_stego_{i}.png'
    
    stego, actual_bpp = embed_lsb_at_target_bpp(
        str(cover_path), 
        target_bpp=0.12
    )
    
    Image.fromarray(stego.astype(np.uint8)).save(stego_path)
    
    print(f"Pair {i}: bpp={actual_bpp:.4f} ✅")

print("=" * 50)
print("Re-embedding complete")
print("Now run unseen_test.py again")