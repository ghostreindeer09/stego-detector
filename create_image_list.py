import pandas as pd
import pathlib

csv_path = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\image_ids_and_rotation.csv'
)

print("Loading CSV... (takes 1-2 minutes)")
df = pd.read_csv(csv_path)

print(f"Total available: {len(df):,} images")
print(f"Columns: {df.columns.tolist()}")

N = 150000
sample = df.sample(n=N, random_state=42)

with open('image_list.txt', 'w') as f:
    for _, row in sample.iterrows():
        image_id = row['ImageID']
        subset   = row.get('Subset', 'train')
        subset   = str(subset).strip().lower()
        if subset not in ['train', 'test',
                          'validation',
                          'challenge2018']:
            subset = 'train'
        f.write(f"{subset}/{image_id}\n")

print(f"Done — image_list.txt created with {N:,} IDs")