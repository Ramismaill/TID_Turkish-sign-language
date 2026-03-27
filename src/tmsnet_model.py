"""
TMS-Net: Temporal Multi-Scale Network for Turkish Sign Language Recognition
6 streams: joint, bone, joint_motion, bone_motion, angle, angle_motion
Multi-scale temporal convolution (fast/medium/slow kernels)
Cross-stream attention fusion
Target: 96%+ on AUTSL (226 classes, isolated signs)
Hardware: RTX 5060 Ti 16GB — batch_size=32 recommended
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import ADJACENCY, NUM_NODES


# ──────────────────────────────────────────────────────
# Spatial Graph Convolution with learnable adjacency
# ──────────────────────────────────────────────────────
class SpatialGCN(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        K = A.shape[0]   # 3 partitions
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.A_learn = nn.Parameter(torch.zeros(K, NUM_NODES, NUM_NODES))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.conv = nn.Conv2d(in_ch * K, out_ch, 1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T, V)
        A = self.A + self.alpha * F.softmax(self.A_learn, dim=-1)
        parts = [torch.einsum('bctv,vw->bctw', x, A[k]) for k in range(A.shape[0])]
        out = self.relu(self.bn(self.conv(torch.cat(parts, dim=1))))
        return out + self.residual(x)


# ──────────────────────────────────────────────────────
# Multi-Scale Temporal Convolution
# ──────────────────────────────────────────────────────
class MultiScaleTCN(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        c = channels // 4    # 4 branches

        self.fast = nn.Sequential(
            nn.Conv2d(channels, c, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.medium = nn.Sequential(
            nn.Conv2d(channels, c, (7, 1), padding=(3, 0)),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.slow = nn.Sequential(
            nn.Conv2d(channels, c, (13, 1), padding=(6, 0)),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.ident = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 4, channels, 1),
            nn.BatchNorm2d(channels)
        )
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.fast(x), self.medium(x), self.slow(x), self.ident(x)], dim=1)
        return self.drop(self.relu(self.fuse(out))) + x


# ──────────────────────────────────────────────────────
# TMS Block = GCN + MultiScaleTCN + optional stride
# ──────────────────────────────────────────────────────
class TMSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, dropout=0.2, stride=1):
        super().__init__()
        self.gcn = SpatialGCN(in_ch, out_ch, A)
        self.tcn = MultiScaleTCN(out_ch, dropout)
        self.stride_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (stride, 1), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        ) if stride > 1 else nn.Identity()

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return self.stride_conv(x)


# ──────────────────────────────────────────────────────
# Per-stream Encoder: (B, 3, T, V) → (B, 256)
# ──────────────────────────────────────────────────────
class StreamEncoder(nn.Module):
    def __init__(self, A, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            TMSBlock(3,   64,  A, dropout),
            TMSBlock(64,  64,  A, dropout),
            TMSBlock(64,  128, A, dropout, stride=2),  # T: 64→32
            TMSBlock(128, 128, A, dropout),
            TMSBlock(128, 256, A, dropout, stride=2),  # T: 32→16
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: (B, 3, T, V)
        return self.pool(self.net(x)).flatten(1)   # (B, 256)


# ──────────────────────────────────────────────────────
# Cross-Stream Attention Fusion
# ──────────────────────────────────────────────────────
class CrossStreamAttention(nn.Module):
    def __init__(self, feat_dim=256, num_streams=6):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * num_streams, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_streams),
            nn.Softmax(dim=-1)
        )

    def forward(self, feats):
        cat = torch.cat(feats, dim=-1)           # (B, 1536)
        w   = self.gate(cat).unsqueeze(-1)       # (B, 6, 1)
        stk = torch.stack(feats, dim=1)          # (B, 6, 256)
        return (w * stk).sum(dim=1)              # (B, 256)


# ──────────────────────────────────────────────────────
# TMS-Net Main Model
# ──────────────────────────────────────────────────────
class TMSNet(nn.Module):
    def __init__(self, num_classes=226, dropout=0.4):
        super().__init__()
        A = ADJACENCY   # (3, 56, 56) numpy array

        self.joint_enc   = StreamEncoder(A, dropout)
        self.bone_enc    = StreamEncoder(A, dropout)
        self.jmotion_enc = StreamEncoder(A, dropout)
        self.bmotion_enc = StreamEncoder(A, dropout)
        self.angle_enc   = StreamEncoder(A, dropout)
        self.amotion_enc = StreamEncoder(A, dropout)

        self.fusion = CrossStreamAttention(256, 6)

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, joint, bone, jmotion, bmotion, angle, amotion):
        # All inputs: (B, 3, T, V)
        feats = [
            self.joint_enc(joint),
            self.bone_enc(bone),
            self.jmotion_enc(jmotion),
            self.bmotion_enc(bmotion),
            self.angle_enc(angle),
            self.amotion_enc(amotion),
        ]
        return self.head(self.fusion(feats))
