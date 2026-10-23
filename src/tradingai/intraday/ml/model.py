"""1D residual / TCN classifier for dip cuts."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for dip-cut models. "
        "Install with: pip install 'tradingai[torch]' or pip install torch"
    ) from exc

from tradingai.intraday.ml.config import (
    DEFAULT_BLOCKS,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN,
    DEFAULT_KERNEL,
    N_SEQ_CHANNELS,
    N_STATIC_FEATURES,
)


def masked_avg_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over time with mask ``[B, T]``; features ``[B, C, T]``."""
    weights = mask.unsqueeze(1).clamp_min(0.0)
    summed = (features * weights).sum(dim=-1)
    denom = weights.sum(dim=-1).clamp_min(1e-6)
    return summed / denom


class ResBlock1d(nn.Module):
    """Two-layer residual conv block with dilation (TCN-style receptive field)."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if out.shape[-1] != residual.shape[-1]:
            t = min(out.shape[-1], residual.shape[-1])
            out = out[..., :t]
            residual = residual[..., :t]
        return self.act(out + residual)


class DipCutClassifier(nn.Module):
    """1D ResNet/TCN encoder + static MLP → 3-class logits.

    Probability order for :meth:`predict_proba` is labels ``(-1, 0, 1)``.
    """

    def __init__(
        self,
        *,
        n_seq_channels: int = N_SEQ_CHANNELS,
        n_static: int = N_STATIC_FEATURES,
        hidden: int = DEFAULT_HIDDEN,
        blocks: int = DEFAULT_BLOCKS,
        kernel: int = DEFAULT_KERNEL,
        dropout: float = DEFAULT_DROPOUT,
        n_classes: int = 3,
    ):
        super().__init__()
        if int(kernel) % 2 == 0:
            raise ValueError("kernel must be odd so padded length is preserved")
        self.stem = nn.Sequential(
            nn.Conv1d(n_seq_channels, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            *[
                ResBlock1d(hidden, kernel, dilation=2**i, dropout=dropout)
                for i in range(int(blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.n_classes = n_classes

    def encode(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.stem(seq)
        h = self.encoder(h)
        return masked_avg_pool(h, mask)

    def forward(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(seq, mask)
        return self.head(torch.cat([pooled, static], dim=-1))

    @torch.no_grad()
    def predict_proba(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(seq, mask, static), dim=-1)


def build_model(**kwargs) -> DipCutClassifier:
    return DipCutClassifier(**kwargs)
