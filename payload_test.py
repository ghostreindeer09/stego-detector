import sys
sys.path.append('.')
import torch
import numpy as np
from PIL import Image
from stego.model import SRNet
import torchvision.transforms as T
from framework.embedding import embed_lsb

ck = torch.load(r'checkpoints\srnet_v31_baseline_best.pth', map_location='cpu')
model = SRNet(num_classes=1, use_kv_hpf=True)
model.load_state_dict(ck['model_state_dict'])
model.eval()

transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
img = Image.open(r'unseen_test/unseen_clean_1.png').convert('RGB')
arr = np.array(img)

for n in [50, 200, 500, 2000, 5000]:
    message = 'A' * n
    stego_arr = embed_lsb(arr, message, algorithm='lsb_sequential', seed=42)
    stego_img = Image.fromarray(stego_arr.astype('uint8'))
    tensor = transform(stego_img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)[0]).item()
    print(f'Payload {n:5d} chars -> STEGO prob: {prob:.4f}')

tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    prob = torch.sigmoid(model(tensor)[0]).item()
print(f'Clean             -> STEGO prob: {prob:.4f}')
