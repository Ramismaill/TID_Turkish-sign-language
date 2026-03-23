import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataset import AUTSLLandmarkDataset
from augmentations import LandmarkAugment
from model import LandmarkSTTransformer, BiLSTMBaseline


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -1.0
        self.count = 0
        self.should_stop = False

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
        return False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            b = x.size(0)
            total_loss += loss.item() * b
            total_acc += accuracy(logits, y) * b
            n += b
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True, help="train_manifest.csv")
    parser.add_argument("--val_csv",      type=str, required=True, help="val_manifest.csv")
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "bilstm"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)  # 4GB VRAM
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = AUTSLLandmarkDataset(args.manifest_csv, augment=LandmarkAugment(target_len=64))
    val_ds   = AUTSLLandmarkDataset(args.val_csv, augment=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = LandmarkSTTransformer(num_classes=226).to(device) \
        if args.model == "transformer" else BiLSTMBaseline(num_classes=226).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1)
    scaler = GradScaler()
    early = EarlyStopping(patience=15)
    best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch - 1 + step / max(1, len(train_loader)))
            b = x.size(0)
            running_loss += loss.item() * b
            running_acc += accuracy(logits.detach(), y) * b
            n += b

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | "
              f"train_loss={running_loss/n:.4f} train_acc={running_acc/n:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early.step(val_acc):
            torch.save({"model": model.state_dict(),
                        "val_acc": val_acc, "epoch": epoch}, best_path)
            print(f"  -> Best saved ({val_acc:.4f})")

        if early.should_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()