"""
stgcn_model.py — Spatial-Temporal Graph Convolutional Network
             for Turkish Sign Language Recognition (AUTSL, 226 classes)

Input tensor shape : (B, 3, 64, 56)   [batch, xyz, frames, nodes]
Output tensor shape: (B, 226)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graph import ADJACENCY, NUM_NODES

NUM_CLASSES = 226


class GraphConv(nn.Module):
    """Single spatial graph convolution (one partition matrix)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)   a: (N, N)
        x = x.permute(0, 2, 3, 1)          # (B, T, N, C)
        x = self.fc(x)                      # (B, T, N, out_ch)
        x = torch.einsum("btnc,mn->btmc", x, a)  # spatial aggregation
        return x.permute(0, 3, 1, 2)        # (B, out_ch, T, N)


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
      Spatial GCN  (3 partitions, summed)
      Temporal CNN (depthwise conv over time)
      Residual connection + BatchNorm + ReLU + Dropout
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self.A = None  # will be set from register_buffer in parent

        # Spatial: one conv per partition
        self.gcn = nn.ModuleList([GraphConv(in_ch, out_ch) for _ in range(3)])
        self.bn_gcn = nn.BatchNorm2d(out_ch)

        # Temporal
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=(9, 1),
                      padding=(4, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )

        # Residual
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Spatial graph conv (sum over 3 partitions)
        out = sum(self.gcn[k](x, A[k]) for k in range(3))
        out = self.bn_gcn(out)

        # Temporal conv + residual
        out = self.relu(self.tcn(out) + self.residual(x))
        return out


class STGCN(nn.Module):
    """
    Full ST-GCN for isolated sign language recognition.

    Architecture (inspired by Yan et al. 2018 + SLR adaptations):
      10 ST-GCN blocks with increasing channel depth
      Global average pooling → FC head
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()

        A = torch.tensor(ADJACENCY, dtype=torch.float32)  # (3, 56, 56)
        self.register_buffer("A", A)

        # Input BN on (joint + velocity) features = 6 channels
        self.data_bn = nn.BatchNorm1d(6 * NUM_NODES)

        # ST-GCN blocks: (in_ch, out_ch, stride)
        cfg = [
            (6,   64,  1),   # 6 input channels: joint(3) + velocity(3)
            (64,  64,  1),
            (64,  64,  1),
            (64,  64,  1),
            (64,  128, 2),   # temporal stride → 64→32 frames
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),   # temporal stride → 32→16 frames
            (256, 256, 1),
            (256, 256, 1),
        ]

        self.blocks = nn.ModuleList([
            STGCNBlock(in_ch, out_ch, stride=s, dropout=dropout)
            for in_ch, out_ch, s in cfg
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, N)
        B, C, T, N = x.shape

        # Input BN: reshape to (B, C*N, T) then back
        x = x.permute(0, 1, 3, 2).contiguous()   # (B, C, N, T)
        x = x.view(B, C * N, T)
        x = self.data_bn(x)
        x = x.view(B, C, N, T).permute(0, 1, 3, 2)  # (B, C, T, N)

        for block in self.blocks:
            x = block(x, self.A)

        return self.head(x)