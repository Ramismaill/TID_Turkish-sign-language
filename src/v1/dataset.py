import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AUTSLLandmarkDataset(Dataset):
    """
    Reads pre-processed .npy files produced by preprocess.py.
    Each .npy file has shape (64, 225): 64 frames × (33 pose + 21 LH + 21 RH) × 3 coords.
    CSV must have columns: npy_path, label
    """
    def __init__(self, csv_path, augment=None):
        self.df = pd.read_csv(csv_path)
        assert {"npy_path", "label"}.issubset(self.df.columns), \
            "CSV must have 'npy_path' and 'label' columns. Run preprocess.py first."
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)  # (64, 225)
        y = int(row["label"])
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
