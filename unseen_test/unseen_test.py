# unseen_test.py
import torch
import pathlib
from PIL import Image
import torchvision.transforms as transforms
import sys
import os  # ✅ for folder checks

# ✅ DEBUG: check folder contents
print("Checking folder contents...")
print(os.listdir(r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'))

sys.path.append(r'C:\Users\KESHAVAREDDY\stego-detector')

from stego.model import SRNet

# Load best checkpoint
checkpoint = torch.load(
    r'C:\Users\KESHAVAREDDY\stego-detector\checkpoints\srnet_v31_baseline_best.pth',
    map_location='cpu'
)

model = SRNet(num_classes=1, use_kv_hpf=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def test_image(image_path, expected_label):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

        # ✅ FIX: handle tuple output
        if isinstance(output, tuple):
            output = output[0]

    prob = torch.sigmoid(output).item()

    prediction = 'STEGO' if prob > 0.5 else 'COVER'
    confidence = prob if prob > 0.5 else 1 - prob
    correct = '✅' if prediction == expected_label else '❌'

    print(f"{correct} [{expected_label}] → {prediction} ({confidence:.2%}) | {pathlib.Path(image_path).name}")

    return prediction == expected_label

results = []

unseen_clean = [
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_clean_1.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_clean_2.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_clean_3.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_clean_4.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_clean_5.png',
]

unseen_stego = [
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_stego_1.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_stego_2.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_stego_3.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_stego_4.png',
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test\unseen_stego_5.png',
]

print("=== CLEAN IMAGES (should predict COVER) ===")
for img in unseen_clean:
    results.append(test_image(img, 'COVER'))

print("\n=== STEGO IMAGES (should predict STEGO) ===")
for img in unseen_stego:
    results.append(test_image(img, 'STEGO'))

accuracy = sum(results) / len(results) * 100
print(f"\n{'='*40}")
print(f"UNSEEN ACCURACY: {accuracy:.1f}%")

if accuracy > 85:
    print("VERDICT: REAL PERFORMANCE ✅ — Model genuinely learned LSB detection")
elif accuracy > 70:
    print("VERDICT: PARTIAL ⚠️ — Some learning but may be overfitting")
else:
    print("VERDICT: OVERFITTING OR LEAKAGE ❌ — Did not generalize")