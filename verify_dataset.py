import pathlib
import random
from PIL import Image
import numpy as np

source = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data\source_images'
)

all_pngs = list(source.rglob('*.png'))
all_jpgs = list(source.rglob('*.jpg'))

print(f"PNG files:  {len(all_pngs):,}")
print(f"JPEG files: {len(all_jpgs):,}")

sample = random.sample(all_pngs, min(100, len(all_pngs)))
corrupt = 0

for path in sample:
    try:
        img = Image.open(path)
        np.array(img)
    except:
        corrupt += 1

print(f"Corrupt:    {corrupt}/100")

if len(all_pngs) >= 50000 and not all_jpgs:
    print("\nVERIFICATION: PASS — Ready for pipeline")
else:
    print("\nVERIFICATION: ISSUES FOUND")
    if len(all_pngs) < 50000:
        print(f"  Need more images: {len(all_pngs):,} < 50,000")
    if all_jpgs:
        print(f"  JPEGs remaining: {len(all_jpgs):,}")
        print(f"  Run convert_to_png.py first")