"""Create minimal ALASKA2-style train/val cover and stego pairs for a quick training run."""
import os
import numpy as np
from PIL import Image

def main():
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ALASKA2")
    for split in ("train", "val"):
        n = 16 if split == "train" else 8
        for folder in ("cover", "stego"):
            d = os.path.join(base, split, folder)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                # Dummy 256x256 RGB image (slightly different for "stego" to simulate payload)
                rng = np.random.default_rng(42 + hash((split, folder, i)) % 1000)
                arr = np.clip(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), 0, 255)
                if folder == "stego":
                    arr = np.clip(arr.astype(np.int32) + rng.integers(-3, 4, arr.shape), 0, 255).astype(np.uint8)
                path = os.path.join(d, f"img_{i:04d}.jpg")
                Image.fromarray(arr).save(path, quality=90)
    print("Created dummy data at", base)

if __name__ == "__main__":
    main()
