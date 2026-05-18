import torch
import torch.nn as nn

NUM_CLASSES = 226


class LandmarkSTTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=64, d_model=256,
                 nhead=8, num_layers=6, ff_dim=1024, dropout=0.4):
        super().__init__()
        self.pose_proj = nn.Linear(99, 96)
        self.lh_proj = nn.Linear(63, 80)
        self.rh_proj = nn.Linear(63, 80)
        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_classes)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        pose = x[:, :, :99]
        lh = x[:, :, 99:162]
        rh = x[:, :, 162:225]
        x = torch.cat([self.pose_proj(pose), self.lh_proj(lh), self.rh_proj(rh)], dim=-1)
        x = self.in_drop(self.in_norm(x))
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, 0])


class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=256, layers=3, dropout=0.4):
        super().__init__()
        self.in_proj = nn.Linear(225, 256)
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, x):
        x = self.in_proj(x)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])