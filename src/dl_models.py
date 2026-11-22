# src/dl_models.py
import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor):
        # h: (B, T, D)
        weights = torch.softmax(self.net(h).squeeze(-1), dim=1)  # (B, T)
        ctx = torch.bmm(weights.unsqueeze(1), h).squeeze(1)      # (B, D)
        return ctx, weights


class ApneaSeqNet(nn.Module):
    def __init__(self, mode: str = "rr_edr", hidden: int = 48, dropout: float = 0.3):
        super().__init__()
        self.mode = mode

        def _conv_block(in_c: int, out_c: int, kernel_size: int, stride: int = 1, drop: float = 0.15) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.GroupNorm(1, out_c),   # LayerNorm over (C, T) — per-sample, subject-agnostic
                    nn.SiLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout1d(drop),     
            )

        
        if mode == "raw_ecg":   
            # 50 Hz (1, 3000) -> ~93 temporal tokens (~1.5 tokens/sec)
            self.stem = nn.Sequential(
                _conv_block(1, 24, kernel_size=15, stride=2, drop=.1),
                _conv_block(24, 48, kernel_size=9, stride=2, drop=.15),
                _conv_block(48, 64, kernel_size=5, stride=2, drop=.15),
            )
        else:
            # 3 Hz (C, 540) -> 67 tokens (~0.37 tokens/sec); (C, 180) -> 22 tokens
            in_ch = 2 if mode == "rr_edr" else 1
            self.stem = nn.Sequential(
                _conv_block(in_ch, 32, kernel_size=7, stride=1, drop=.1),
                _conv_block(32, 64, kernel_size=5, stride=1, drop=.15),
                _conv_block(64, 64, kernel_size=5, stride=1, drop=.15),
            )

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden,
            num_layers=2, # Increasing hidden layers doesn't decide as much as increasing hidden size and bidirectionality
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = TemporalAttentionPool(hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = self.stem(x).transpose(1, 2)   # (B, T, 64)
        h, _ = self.gru(h)                 # (B, T, 2*hidden)
        ctx, weights = self.attn(h)        # (B, 2*hidden), (B, T)
        logits = self.head(ctx).squeeze(-1)
        return (logits, weights) if return_attn else logits