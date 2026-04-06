import torch
import pathlib
from PIL import Image
import torchvision.transforms as transforms
from stego.model import SRNet

def build_model():
    model = SRNet()
    return model

checkpoint = torch.load(
    r'C:\Users\KESHAVAREDDY\stego-detector\checkpoints\srnet_v31_baseline_best.pth',
    map_location='cpu',
    weights_only=False
)

model = build_model()

if 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])

model.eval()
print("Model loaded successfully")

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image_path, true_label):
    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        
        # Handle tuple output
        if isinstance(output, tuple):
            output = output[0]
        
        # Handle different output shapes
        if output.dim() > 1:
            output = output.squeeze()
        
        prob = torch.sigmoid(output).item()
    
    prediction = 'STEGO' if prob > 0.5 else 'COVER'
    confidence = prob if prob > 0.5 else 1 - prob
    correct    = prediction == true_label
    status     = 'CORRECT' if correct else 'WRONG'
    
    print(f"{status} | {pathlib.Path(image_path).name}")
    print(f"  True:       {true_label}")
    print(f"  Predicted:  {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print()
    return correct

unseen_dir = pathlib.Path(
    r'C:\Users\KESHAVAREDDY\stego-detector\unseen_test'
)

print("UNSEEN IMAGE TEST — 256x256 IMAGES")
print("=" * 50)
print("CLEAN IMAGES (should predict COVER):")
print("-" * 50)

results = []
for i in range(1, 6):
    path   = unseen_dir / f'unseen_clean_{i}_256.png'
    result = predict(str(path), 'COVER')
    results.append(result)

print("STEGO IMAGES (should predict STEGO):")
print("-" * 50)

for i in range(1, 6):
    path   = unseen_dir / f'unseen_stego_{i}_256.png'
    result = predict(str(path), 'STEGO')
    results.append(result)

correct  = sum(results)
total    = len(results)
accuracy = correct / total * 100

print("=" * 50)
print(f"Correct:  {correct}/{total}")
print(f"Accuracy: {accuracy:.1f}%")
print()

if accuracy >= 85:
    print("VERDICT: GENUINELY GOOD MODEL ✅")
    print("Model learned real LSB detection")
elif accuracy >= 70:
    print("VERDICT: PARTIAL LEARNING ⚠️")
    print("Some generalization but needs improvement")
else:
    print("VERDICT: NOT GENERALIZING ❌")
    print("Further investigation needed")
