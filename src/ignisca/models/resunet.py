from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 conv + GroupNorm + ReLU with an additive skip and 2D spatial dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.skip: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = torch.relu(out + residual)
        return self.dropout(out)


class ResUNet(nn.Module):
    """ResU-Net for next-timestep fire perimeter prediction.

    Input: (B, in_channels, H, W) — IgnisCA feature stack, 12 channels.
    Output: (B, 1, H, W) raw logits (sigmoid is applied in the loss / at inference).
    """

    def __init__(self, in_channels: int = 12, base: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = ResidualBlock(in_channels, c1, dropout=dropout)
        self.enc2 = ResidualBlock(c1, c2, dropout=dropout)
        self.enc3 = ResidualBlock(c2, c3, dropout=dropout)
        self.enc4 = ResidualBlock(c3, c4, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(c4, c4, dropout=dropout),
            ResidualBlock(c4, c4, dropout=dropout),
        )

        self.up4 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(c4 + c4, c4, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(c3 + c3, c3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(c2 + c2, c2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(c1 + c1, c1, dropout=dropout)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)
