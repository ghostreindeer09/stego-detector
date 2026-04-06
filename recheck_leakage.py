import pathlib
import hashlib
from PIL import Image
import numpy as np

data_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data'
)

print("RUNNING FINAL LEAKAGE DETECTION...")
print("=" * 50)

# Collect hashes for each split
splits = {}
for split in ['train', 'val', 'test']:
    cover_dir = data_dir / split / 'cover'
    paths     = list(cover_dir.rglob('*.png'))
    hashes    = set(
        hashlib.md5(open(p, 'rb').read()).hexdigest()
        for p in paths
    )
    splits[split] = {
        'paths':  paths,
        'hashes': hashes
    }
    print(f"{split}: {len(paths)} images hashed")

# CHECK 1: HASH OVERLAP
print("\nCHECK 1: HASH OVERLAP")
passed = True
for a, b in [('train', 'val'),
             ('train', 'test'),
             ('val',   'test')]:
    overlap = splits[a]['hashes'] & \
              splits[b]['hashes']
    if len(overlap) == 0:
        print(f"  {a}/{b}: PASS ✅")
    else:
        print(f"  {a}/{b}: FAIL ❌ "
              f"{len(overlap)} overlapping images")
        passed = False

# CHECK 2: PIXEL IDENTITY
print("\nCHECK 2: PIXEL IDENTITY")
def pixel_hash(path):
    img = np.array(Image.open(path))
    return hashlib.md5(img.tobytes()).hexdigest()

for a, b in [('train', 'val'),
             ('train', 'test'),
             ('val',   'test')]:
    ha = set(
        pixel_hash(p) 
        for p in splits[a]['paths'][:100]
    )
    hb = set(
        pixel_hash(p) 
        for p in splits[b]['paths'][:100]
    )
    overlap = ha & hb
    if len(overlap) == 0:
        print(f"  {a}/{b}: PASS ✅")
    else:
        print(f"  {a}/{b}: FAIL ❌ "
              f"{len(overlap)} pixel-identical images")
        passed = False

# CHECK 3: PAIR ISOLATION
print("\nCHECK 3: PAIR ISOLATION")
for split_a in ['train', 'val', 'test']:
    stego_dir    = data_dir / split_a / 'stego'
    stego_hashes = set(
        hashlib.md5(open(p, 'rb').read()).hexdigest()
        for p in stego_dir.rglob('*.png')
    )
    for split_b in ['train', 'val', 'test']:
        if split_a == split_b:
            continue
        cross = stego_hashes & splits[split_b]['hashes']
        if len(cross) == 0:
            print(f"  stego/{split_a} vs cover/{split_b}: PASS ✅")
        else:
            print(f"  stego/{split_a} vs cover/{split_b}: FAIL ❌ "
                  f"{len(cross)} cross contaminated")
            passed = False

# FINAL VERDICT
print("\n" + "=" * 50)
if passed:
    print("LEAKAGE CHECK VERDICT: CLEAN ✅")
    print("No contamination detected")
else:
    print("LEAKAGE CHECK VERDICT: CONTAMINATED ❌")
    print("Leakage survived — investigate before trusting results")