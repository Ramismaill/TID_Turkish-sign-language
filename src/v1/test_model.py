import sys
sys.path.insert(0, 'src')

import torch
import json
from stgcn_model import STGCN

device = 'cuda'
model = STGCN(num_classes=226, dropout=0.5).to(device)
ckpt = torch.load('C:/AUTSL_project/checkpoints/best_stgcn.pt', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

with open('C:/AUTSL_project/src/class_map.json', encoding='utf-8') as f:
    cm = json.load(f)

# Test with random input
x = torch.randn(1, 6, 64, 56).to(device)
with torch.no_grad():
    out = torch.softmax(model(x), dim=1)
    conf, idx = torch.max(out, 1)
    print(f'Model works.')
    print(f'Random prediction: {cm[str(idx.item())]} ({conf.item()*100:.1f}%)')
    print(f'Total classes: {len(cm)}')
