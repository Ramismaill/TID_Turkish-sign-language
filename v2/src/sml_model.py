"""
sml_model.py — Skeleton-based Multi-feature Learning (SML)
               3-Stream GCN with Cross-Stream Attention Fusion

Key difference from CTR-GCN:
  - CTR-GCN: 4 independent streams -> concat -> FC (no cross-stream communication)
  - SML: 3 stream encoders + cross-stream attention (streams inform each other)
  - SML: Streams pre-computed in dataset (bone/motion from augmented joints)

3 streams:
  1. Joint    — absolute position (x, y, z)
  2. Bone     — child - parent (relative structure)
  3. Motion   — frame_t - frame_t-1 (velocity)

Input:  3 × (B, 3, 64, 56)
Output: (B, 226)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import ADJACENCY, NUM_NODES, BONE_PAIRS

NUM_CLASSES = 226


# ── Temporal Convolution (multi-scale, reused from CTR-GCN pattern) ──────────

class TemporalConv(nn.Module):
    """3-branch multi-scale temporal convolution: k=3, k=5, maxpool."""

    def __init__(self, ch, stride=1, dropout=0.3):
        super().__init__()
        b1 = ch // 3
        b2 = ch // 3
        b3 = ch - b1 - b2

        self.b1 = nn.Sequential(
            nn.Conv2d(ch, b1, (3, 1), padding=(1, 0), stride=(stride, 1)),
            nn.BatchNorm2d(b1), nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(ch, b2, (5, 1), padding=(2, 0), stride=(stride, 1)),
            nn.BatchNorm2d(b2), nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), padding=(1, 0), stride=(stride, 1)),
            nn.Conv2d(ch, b3, 1),
            nn.BatchNorm2d(b3), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


# ── Spatial Graph Convolution (lightweight, no dynamic topology) ─────────────

class SpatialGCN(nn.Module):
    """3-partition graph convolution (self, centripetal, centrifugal).
    Simpler than CTRConv — no dynamic graph attention, keeping SML lightweight."""

    def __init__(self, in_ch, out_ch, num_partitions=3):
        super().__init__()
        self.num_partitions = num_partitions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            for _ in range(num_partitions)
        ])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: (B, C, T, V)  A: (3, V, V)
        out = sum(
            torch.einsum('bctv,vw->bctw', self.convs[k](x), A[k])
            for k in range(self.num_partitions)
        )
        return self.relu(self.bn(out))


# ── SML Block (Spatial GCN + Temporal Conv + Residual) ───────────────────────

class SMLBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.gcn = SpatialGCN(in_ch, out_ch)
        self.tcn = TemporalConv(out_ch, stride=stride, dropout=dropout)

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        return self.relu(self.tcn(self.gcn(x, A)) + self.residual(x))


# ── Per-Stream Encoder (8 blocks, lighter than CTR-GCN's 10) ────────────────

class SMLStreamEncoder(nn.Module):
    """Single stream GCN encoder: (B, 3, 64, 56) -> (B, 256)"""

    def __init__(self, in_channels=3, dropout=0.3):
        super().__init__()

        A = torch.tensor(ADJACENCY, dtype=torch.float32)
        self.register_buffer("A", A)
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)

        cfg = [
            (in_channels, 64,  1),
            (64,  64,  1),
            (64,  64,  1),
            (64,  128, 2),     # temporal stride
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),     # temporal stride
            (256, 256, 1),
        ]

        self.blocks = nn.ModuleList([
            SMLBlock(ic, oc, stride=s, dropout=dropout)
            for ic, oc, s in cfg
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        B, C, T, V = x.shape
        # Data BatchNorm
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2)  # (B, C, T, V)

        for block in self.blocks:
            x = block(x, self.A)

        return self.pool(x).view(B, -1)  # (B, 256)


# ── Cross-Stream Attention ───────────────────────────────────────────────────

class CrossStreamAttention(nn.Module):
    """Multi-head self-attention across stream embeddings.
    Allows joint/bone/motion streams to inform each other."""

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 3, 256) — 3 stream tokens
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x  # (B, 3, 256)


# ── SML Top-Level Model ─────────────────────────────────────────────────────

class SML(nn.Module):
    """
    Skeleton-based Multi-feature Learning (SML)

    3 stream GCN encoders + 2-layer cross-stream attention + classification head.
    Input:  joint (B,3,64,56), bone (B,3,64,56), motion (B,3,64,56)
    Output: (B, 226)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()

        self.joint_stream  = SMLStreamEncoder(in_channels=3, dropout=dropout)
        self.bone_stream   = SMLStreamEncoder(in_channels=3, dropout=dropout)
        self.motion_stream = SMLStreamEncoder(in_channels=3, dropout=dropout)

        self.cross_attn = nn.Sequential(
            CrossStreamAttention(embed_dim=256, num_heads=4, dropout=0.1),
            CrossStreamAttention(embed_dim=256, num_heads=4, dropout=0.1),
        )

        self.head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, joint, bone, motion):
        # Each: (B, 3, 64, 56)
        j = self.joint_stream(joint)    # (B, 256)
        b = self.bone_stream(bone)      # (B, 256)
        m = self.motion_stream(motion)  # (B, 256)

        # Stack into sequence of 3 tokens and apply cross-stream attention
        tokens = torch.stack([j, b, m], dim=1)  # (B, 3, 256)
        tokens = self.cross_attn(tokens)          # (B, 3, 256)

        # Flatten and classify
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 768)
        return self.head(fused)
