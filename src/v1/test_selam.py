import sys
sys.path.insert(0, 'C:/AUTSL_project/src')

import torch
import numpy as np
import pandas as pd
from stgcn_model import STGCN
from graph import KEEP_INDICES, NUM_NODES

device = 'cuda'
model = STGCN(num_classes=226, dropout=0.5).to(device)
ckpt = torch.load('C:/AUTSL_project/checkpoints/best_stgcn.pt',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

df = pd.read_csv('C:/AUTSL_project/landmarks/train_manifest.csv')

# Test 5 real samples of "selam" (label 173)
samples = df[df['label'] == 173].head(5)
print("Testing 5 real 'selam' samples from training data:")
print("-" * 50)

correct = 0
for i, (_, row) in enumerate(samples.iterrows()):
    x = np.load(row['npy_path']).astype('float32')
    x = x[:, KEEP_INDICES].reshape(64, NUM_NODES, 3).transpose(2, 0, 1)
    x_t = torch.tensor(x).unsqueeze(0).to(device)
    vel = torch.zeros_like(x_t)
    vel[:, :, 1:, :] = x_t[:, :, 1:, :] - x_t[:, :, :-1, :]
    x_t = torch.cat([x_t, vel * 10], dim=1)

    with torch.no_grad():
        out = torch.softmax(model(x_t), dim=1)
        conf, idx = torch.max(out, 1)
        predicted = idx.item()
        is_correct = predicted == 173
        if is_correct:
            correct += 1
        print(f"Sample {i+1}: predicted={predicted} ({'selam' if is_correct else 'WRONG'}) conf={conf.item()*100:.1f}%")

print("-" * 50)
print(f"Accuracy on selam: {correct}/5 correct")
