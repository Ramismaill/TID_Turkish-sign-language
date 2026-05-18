import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from sml_model import SML
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = SML(num_classes=226, dropout=0.3).to(device)
ckpt = torch.load('C:/AUTSL_project/checkpoints/best_sml.pt',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"SML loaded OK -- Best val_acc: {ckpt.get('val_acc', 'N/A')}")

BONE_SRC = [p[0] for p in BONE_PAIRS]
BONE_DST = [p[1] for p in BONE_PAIRS]

df = pd.read_csv('C:/AUTSL_project/landmarks/val_manifest.csv')
samples = df.head(5)

print("\nTesting 5 real samples:")
print("-" * 50)
correct = 0
for i, (_, row) in enumerate(samples.iterrows()):
    x = np.load(row['npy_path']).astype('float32')
    x = x[:, KEEP_INDICES].reshape(64, NUM_NODES, 3).transpose(2, 0, 1)
    joint  = torch.tensor(x).unsqueeze(0).to(device)
    bone   = joint[:, :, :, BONE_SRC] - joint[:, :, :, BONE_DST]
    motion = torch.zeros_like(joint)
    motion[:, :, 1:, :] = joint[:, :, 1:, :] - joint[:, :, :-1, :]

    with torch.no_grad():
        out  = torch.softmax(model(joint, bone, motion), dim=1)
        conf, idx = torch.max(out, 1)
        predicted = idx.item()
        is_correct = predicted == int(row['label'])
        if is_correct:
            correct += 1
        print(f"Sample {i+1}: true={int(row['label'])} predicted={predicted} "
              f"{'OK' if is_correct else 'WRONG'} conf={conf.item()*100:.1f}%")

print("-" * 50)
print(f"Accuracy: {correct}/5")
print("SML inference pipeline ready")