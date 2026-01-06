# src/model/attention.py
"""
Attention modules for RISRANet (balanced: readable + optimized).

Contains:
- channel_shuffle(x, groups)
- ShuffleAttention: the SA module used inside CSA/DSA blocks
- ConvolutionalShuffleAttention (CSA)
- DenseShuffleAttention (DSA)  -- a denser conv stack + SA
- MultiScaleShuffleAttention (MSA) -- parallel dilated convs + SA/agg

Design goals:
- Clear, small building blocks
- Avoid Python-level loops over channels (use grouped ops)
- Use nn.Sequential for compact small nets
- Docstrings + simple defaults so you can tweak hyperparams easily

Test:
> python -m src.model.attention
should print shapes and run a smoke forward pass.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helper: channel shuffle
# -----------------------------
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle for tensor (B, C, H, W).
    Splits channels into `groups` groups and shuffles across groups.
    This is the standard ShuffleNet channel shuffle.
    """
    b, c, h, w = x.shape
    if groups == 1 or c % groups != 0:
        return x
    channels_per_group = c // groups
    # reshape -> (b, groups, channels_per_group, h, w)
    x = x.view(b, groups, channels_per_group, h, w)
    # transpose groups and channels_per_group -> (b, channels_per_group, groups, h, w)
    x = x.transpose(1, 2).contiguous()
    # flatten back
    x = x.view(b, c, h, w)
    return x


# -----------------------------
# Shuffle Attention (SA)
# -----------------------------
class ShuffleAttention(nn.Module):
    """
    Shuffle Attention module:
    - splits input channels into groups
    - each group is split into two branches: channel-attention branch and spatial-attention branch
    - outputs are concatenated per group and channel-shuffled across groups
    """

    def __init__(self, in_channels: int, groups: int = 8, reduction: int = 16):
        """
        in_channels: total channels
        groups: number of channel groups (G in paper)
        reduction: channel reduction for channel-attention MLP
        """
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.in_channels = in_channels
        self.groups = groups
        self.sub_ch = in_channels // groups  # channels per group

        # channel-attention: simple SE-like MLP (applied to half sub-ch)
        self.ca_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.sub_ch // 2, max(1, (self.sub_ch // 2) // reduction), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, (self.sub_ch // 2) // reduction), self.sub_ch // 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # spatial-attention branch uses group norm + 1x1 conv gating
        self.spatial_gn = nn.GroupNorm(1, self.sub_ch // 2)  # treat spatial branch channels as group of 1
        self.spatial_fc = nn.Sequential(
            nn.Conv2d(self.sub_ch // 2, self.sub_ch // 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        G = self.groups
        assert c == self.in_channels
        # split channels into G groups -> (B, G, sub_ch, H, W) then iterate via reshape transforms
        x = x.view(b, G, self.sub_ch, h, w)

        outs = []
        for g in range(G):
            xg = x[:, g]  # (B, sub_ch, H, W)
            # split the group channels into two equal branches along channel dim
            c_half = self.sub_ch // 2
            xg1 = xg[:, :c_half, :, :]  # channel-att
            xg2 = xg[:, c_half:, :, :]  # spatial-att

            # Channel attention branch
            ca = self.ca_fc(xg1)  # (B, c_half, 1,1)
            xg1_att = xg1 * ca

            # Spatial attention branch
            s = self.spatial_gn(xg2)
            s = self.spatial_fc(s)  # (B, c_half, H, W)
            xg2_att = xg2 * s

            # concat back
            xg_out = torch.cat([xg1_att, xg2_att], dim=1)  # (B, sub_ch, H, W)
            outs.append(xg_out)

        # stack groups back -> (B, G, sub_ch, H, W) then flatten to (B, C, H, W)
        x_out = torch.stack(outs, dim=1).view(b, c, h, w)

        # channel shuffle to mix group info
        x_out = channel_shuffle(x_out, G)
        return x_out


# -----------------------------
# Convolutional Shuffle Attention (CSA)
# -----------------------------
class ConvolutionalShuffleAttention(nn.Module):
    """
    CSA module:
    - 1x1 conv to reduce dimension
    - 3x3 conv
    - ShuffleAttention
    - final 1x1 conv to project back
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32, groups: int = 8):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.sa = ShuffleAttention(hidden_channels, groups=groups)
        self.project = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        y = self.activation(self.conv(y))
        y = self.sa(y)
        y = self.project(y)
        return y


# -----------------------------
# Dense Shuffle Attention (DSA)
# -----------------------------
class DenseShuffleAttention(nn.Module):
    """
    DSA: a denser variant that stacks multiple conv blocks with concatenation (like dense connection).
    This is a compact, efficient approximation of the paper's DSA block.
    """

    def __init__(self, in_channels: int, growth: int = 32, layers: int = 4, groups: int = 8):
        super().__init__()
        self.layers = layers
        self.growth = growth
        self.in_channels = in_channels

        # initial conv to growth channels
        self.initial = nn.Conv2d(in_channels, growth, kernel_size=3, padding=1, bias=True)

        # successive convs that take concatenated input
        self.blocks = nn.ModuleList()
        for i in range(layers - 1):
            in_ch = in_channels + growth * (i + 1)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ))

        # SA over concatenated features (project down to in_channels then SA)
        total_ch = in_channels + growth * layers
        self.project = nn.Conv2d(total_ch, in_channels, kernel_size=1, bias=True)
        self.sa = ShuffleAttention(in_channels, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        out = self.initial(x)
        feats.append(out)
        for blk in self.blocks:
            inp = torch.cat(feats, dim=1)
            out = blk(inp)
            feats.append(out)
        cat = torch.cat(feats, dim=1)
        proj = self.project(cat)
        att = self.sa(proj)
        return att


# -----------------------------
# Multi-Scale Shuffle Attention (MSA)
# -----------------------------
class MultiScaleShuffleAttention(nn.Module):
    """
    MSA: parallel dilated convolution branches (dilation rates list),
    followed by concat -> SA -> project back. Dilated convs increase receptive field.
    """

    def __init__(self, in_channels: int, mid_channels: int = 32, dilation_rates: Optional[List[int]] = None, groups: int = 8):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 5]
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            # keep padding = dilation to keep spatial size
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=d, dilation=d, bias=True),
                    nn.ReLU(inplace=True),
                )
            )
        total_mid = mid_channels * len(dilation_rates)
        self.project = nn.Conv2d(total_mid, in_channels, kernel_size=1, bias=True)
        self.sa = ShuffleAttention(in_channels, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for br in self.branches:
            outs.append(br(x))
        cat = torch.cat(outs, dim=1)
        proj = self.project(cat)
        att = self.sa(proj)
        return att


# -----------------------------
# Small smoke test
# -----------------------------
def _smoke_test():
    print("Running attention module smoke test...")
    b, c, h, w = 2, 32, 64, 64
    x = torch.randn(b, c, h, w)
    print("input:", x.shape)

    csa = ConvolutionalShuffleAttention(in_channels=c, hidden_channels=32, groups=8)
    dsa = DenseShuffleAttention(in_channels=c, growth=16, layers=4, groups=8)
    msa = MultiScaleShuffleAttention(in_channels=c, mid_channels=16, dilation_rates=[1, 2, 5], groups=8)

    for name, module in [("CSA", csa), ("DSA", dsa), ("MSA", msa)]:
        y = module(x)
        print(f"{name} output:", y.shape)
    print("Smoke test done.")


if __name__ == "__main__":
    _smoke_test()
