import numpy as np
from PIL import Image
import pathlib
import random

def embed_lsb(image_path, target_bpp=0.12):
    # Load and resize to 256x256 FIRST
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.LANCZOS)
    img_array = np.array(img).astype(int)
    
    total    = img_array.size
    n_bits   = int(total * target_bpp)
    msg_bits = np.random.randint(0, 2, n_bits)
    flat     = img_array.flatten().copy()
    positions = random.sample(range(len(flat)), n_bits)
    
    for idx, pos in enumerate(positions):
        flat[pos] = (flat[pos] & 0xFE) | msg_bits[idx]
    
    stego = flat.reshape(256, 256, 3)
    diff  = np.sum(img_array != stego)
    bpp   = diff / total
    
    return img, stego, bpp

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)

print("RE-EMBEDDING AT 256x256 TO MATCH TRAINING")
print("=" * 50)

for i in range(1, 6):
    cover_path = unseen_dir / f'unseen_clean_{i}.png'
    stego_path = unseen_dir / f'unseen_stego_{i}.png'
    cover_256_path = unseen_dir / f'unseen_clean_{i}_256.png'
    stego_256_path = unseen_dir / f'unseen_stego_{i}_256.png'
    
    cover_img, stego_arr, bpp = embed_lsb(
        str(cover_path), target_bpp=0.12
    )
    
    # Save 256x256 versions
    cover_img.save(cover_256_path)
    Image.fromarray(
        stego_arr.astype(np.uint8)
    ).save(stego_256_path)
    
    print(f"Pair {i}: bpp={bpp:.4f} ✅")
    print(f"  Cover saved: {cover_256_path.name}")
    print(f"  Stego saved: {stego_256_path.name}")

print("=" * 50)
print("Done — now run unseen_test_256.py")
