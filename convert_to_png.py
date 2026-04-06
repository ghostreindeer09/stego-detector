import pathlib
from PIL import Image
from tqdm import tqdm
import shutil

source = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data\source_images'
)

files = (
    list(source.rglob('*.jpg')) +
    list(source.rglob('*.jpeg'))
)
print(f"Found {len(files):,} JPEG files to convert")
print(f"Estimated time: {len(files)//300} minutes")

success = 0
failed  = 0

for jpg in tqdm(files):
    try:
        png = jpg.with_suffix('.png')
        img = Image.open(jpg).convert('RGB')
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        img.save(png, format='PNG')
        jpg.unlink()
        success += 1
    except Exception as e:
        failed += 1

total, used, free = shutil.disk_usage("C:\\")
print(f"\nConversion Complete:")
print(f"  Converted: {success:,}")
print(f"  Failed:    {failed:,}")
print(f"  Free disk: {free/1e9:.1f} GB")