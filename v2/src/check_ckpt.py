import sys
sys.path.insert(0, 'src')
import torch

files = ['best_stgcn.pt', 'best_sml.pt', 'best_transformer.pt', 'resume_stgcn.pt', 'resume_sml.pt']

print("=" * 55)
print(f"{'File':<25} {'Val Acc':>10} {'Epoch':>8}")
print("=" * 55)

for name in files:
    try:
        ckpt = torch.load(f'C:/AUTSL_project/checkpoints/{name}',
                          map_location='cpu', weights_only=False)
        val_acc = ckpt.get('val_acc', 'N/A')
        epoch   = ckpt.get('epoch', 'N/A')
        if isinstance(val_acc, float):
            val_acc = f"{val_acc*100:.2f}%"
        print(f"{name:<25} {val_acc:>10} {str(epoch):>8}")
    except Exception as e:
        print(f"{name:<25} {'ERROR':>10}")

print("=" * 55)