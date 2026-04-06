from PIL import Image
import pathlib
import random

cover_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data\splits_v31\train\cover'
)
covers = sorted(cover_dir.rglob('*.png'))
sample = random.sample(covers, 5)

print("Training image sizes:")
for p in sample:
    img = Image.open(p)
    print(f"  {p.name}: {img.size}")

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)
print("\nUnseen test image sizes:")
for i in range(1, 6):
    img = Image.open(unseen_dir / f'unseen_clean_{i}.png')
    print(f"  unseen_clean_{i}.png: {img.size}")
