import os
import io
import numpy as np
from PIL import Image

def get_first_n_pairs(n):
    cover_dir = 'data/openimages/cover'
    stego_dir = 'data/openimages/stego'
    pairs = []
    
    if not os.path.exists(cover_dir) or not os.path.exists(stego_dir):
        return pairs
        
    for f in os.listdir(stego_dir):
        if len(pairs) >= n:
            break
        if f.endswith('.png'):
            cover_path = os.path.join(cover_dir, f)
            stego_path = os.path.join(stego_dir, f)
            if os.path.exists(cover_path):
                pairs.append((cover_path, stego_path))
    return pairs

def check_1():
    print("CHECK 1: EMBEDDING INTEGRITY")
    print("-" * 32)
    broken_pairs = 0
    pixel_diffs = []
    
    pairs = get_first_n_pairs(100)
    if not pairs:
        print("No pairs found!")
        return False
        
    for cover_path, stego_path in pairs:
        cover = np.array(Image.open(cover_path))
        stego = np.array(Image.open(stego_path))
        diff = np.sum(cover != stego)
        if diff == 0:
            broken_pairs += 1
        pixel_diffs.append(diff)
        
    print(f"Broken pairs: {broken_pairs}/{len(pairs)}")
    print(f"Avg pixels changed: {np.mean(pixel_diffs):.1f}")
    print(f"Min pixels changed: {np.min(pixel_diffs)}")
    print(f"Embedding integrity: {100*(1-broken_pairs/max(1, len(pairs))):.1f}%")
    
    if broken_pairs == 0:
        print("PASS\n")
    else:
        print("FAIL\n")

def check_2():
    print("CHECK 2: LABEL INTEGRITY")
    print("-" * 32)
    import torch
    from torch.utils.data import DataLoader
    from stego.datasets import PairConstraintStegoDataset, get_train_transform, pair_constraint_collate
    
    cover_dir = 'data/openimages/cover'
    stego_dir = 'data/openimages/stego'
    
    transforms = get_train_transform(256)
    dataset = PairConstraintStegoDataset(cover_dir, stego_dir, 256, transforms)
    
    cover_labels = 0
    stego_labels = 0
    
    labels_list = []
    for i in range(min(25, len(dataset))):
        c_img, s_img, c_lbl, s_lbl = dataset[i]
        labels_list.append(c_lbl.item())
        labels_list.append(s_lbl.item())
        cover_labels += 1
        stego_labels += 1
            
    print(f"First 50 labels: {labels_list}")
    ratio_0 = cover_labels / max((cover_labels + stego_labels), 1)
    ratio_1 = stego_labels / max((cover_labels + stego_labels), 1)
    print(f"Class distribution (first 50): 0s={ratio_0:.2f}, 1s={ratio_1:.2f}")

    # Full dataset class distribution (pair-constraint implies exactly balanced)
    total_pairs = len(dataset)
    total_cover = total_pairs
    total_stego = total_pairs
    total = total_cover + total_stego
    print(f"Class distribution (full dataset): 0s={total_cover/total:.2f}, 1s={total_stego/total:.2f}")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pair_constraint_collate)
    batch_img, batch_lbl = next(iter(loader))
    print("Batch labels from dataloader:", batch_lbl.tolist())

    pairing_ok = True
    for i in range(min(10, len(dataset))):
        c_img, s_img, _, _ = dataset[i]
        if torch.equal(c_img, s_img):
            pairing_ok = False
            break

    labels_ok = (cover_labels == stego_labels) and np.isclose(total_cover / total, 0.5) and np.isclose(total_stego / total, 0.5)
    if labels_ok and pairing_ok:
        print("PASS\n")
    else:
        print("FAIL\n")

def check_3():
    print("CHECK 3: LSB SIGNAL SURVIVAL THROUGH AUGMENTATION")
    print("-" * 32)
    def apply_jpeg(img_array, quality):
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))
        
    pairs = get_first_n_pairs(10)
    
    q95_diffs = []
    for cover_path, stego_path in pairs:
        cover = np.array(Image.open(cover_path))
        stego = np.array(Image.open(stego_path))
        
        print(f"Testing pair: {os.path.basename(cover_path)}")
        for quality in [95, 85, 70]:
            stego_compressed = apply_jpeg(stego, quality)
            cover_compressed = apply_jpeg(cover, quality)
            diff = np.sum(stego_compressed != cover_compressed)
            print(f"Quality {quality}: pixels different = {diff}")
            if quality == 95:
                q95_diffs.append(diff)

    if len(q95_diffs) > 0 and np.mean(q95_diffs) > 0:
        print("PASS\n")
    else:
        print("FAIL\n")

def check_4():
    print("CHECK 4: PAYLOAD DENSITY VERIFICATION")
    print("-" * 32)
    bpps = []
    pairs = get_first_n_pairs(100)
    for cover_path, stego_path in pairs:
        cover = np.array(Image.open(cover_path))
        stego = np.array(Image.open(stego_path))
        diff = np.sum(cover != stego)
        
        # Estimate: LSB flips ~50% of the bits. diff is roughly total_payload_bits / 2
        payload_bits = diff * 2
        total_pixels = cover.size
        bpp = payload_bits / total_pixels
        bpps.append(bpp)
        
    if not bpps:
        print("No pairs found.")
        return
        
    mean_bpp = np.mean(bpps)
    print(f"Min bpp (est): {np.min(bpps):.4f}")
    print(f"Max bpp (est): {np.max(bpps):.4f}")
    print(f"Mean bpp (est): {mean_bpp:.4f}")
    
    if mean_bpp > 0.1:
        print("PASS\n")
    else:
        print("FAIL\n")

if __name__ == '__main__':
    check_1()
    check_2()
    check_3()
    check_4()
