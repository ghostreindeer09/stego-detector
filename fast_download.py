import pathlib
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Read your image list
with open('image_list.txt', 'r') as f:
    lines = f.readlines()

print(f"Total images to download: {len(lines):,}")

output_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\data\source_images'
)
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV for thumbnail URLs (faster to download)
print("Loading CSV for thumbnail URLs...")
df = pd.read_csv(
    r'C:\Users\KESHAVAREDDY\stego-detector\image_ids_and_rotation.csv'
)
df = df.set_index('ImageID')
print(f"CSV loaded: {len(df):,} entries")

def download_image(image_id):
    output_path = output_dir / f"{image_id}.jpg"
    
    # Skip if already downloaded
    if output_path.exists():
        return 'skipped'
    
    try:
        # Use thumbnail URL — much faster than full size
        # Thumbnails are ~300KB vs ~2MB for originals
        row = df.loc[image_id]
        
        # Try thumbnail first
        url = row.get('Thumbnail300KURL', '')
        if not url or pd.isna(url):
            url = row.get('OriginalURL', '')
        
        if not url or pd.isna(url):
            return 'no_url'
        
        response = requests.get(
            url, timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return 'success'
        else:
            return f'error_{response.status_code}'
            
    except Exception as e:
        return f'failed'

# Extract just image IDs
image_ids = []
for line in lines:
    line = line.strip()
    if '/' in line:
        image_id = line.split('/')[-1]
    else:
        image_id = line
    image_ids.append(image_id)

print(f"\nStarting fast download with 30 threads...")
print(f"Output: {output_dir}")

success = 0
skipped = 0
failed  = 0

with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {
        executor.submit(download_image, img_id): img_id 
        for img_id in image_ids
    }
    
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            result = future.result()
            if result == 'success':
                success += 1
            elif result == 'skipped':
                skipped += 1
            else:
                failed += 1
            
            pbar.set_postfix({
                'ok': success,
                'skip': skipped,
                'fail': failed
            })
            pbar.update(1)

# Count final result
all_files = list(output_dir.rglob('*.jpg'))
print(f"\nDownload Complete:")
print(f"  Downloaded: {success:,}")
print(f"  Skipped:    {skipped:,}")
print(f"  Failed:     {failed:,}")
print(f"  Total files:{len(all_files):,}")