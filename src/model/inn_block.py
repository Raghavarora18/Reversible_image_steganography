# src/model/inn_block.py
"""
Invertible affine coupling block used in RISRANet.

Forward (hiding):
    y1 = x1 + Phi(x2)
    y2 = eta(y1) + x2 * exp( sigma( rho(y1) ) )

Inverse (recovery):
    x2 = (y2 - eta(y1)) * exp( -sigma( rho(y1) ) )
    x1 = y1 - Phi(x2)

Phi, rho, eta are small neural nets. We use attention modules where appropriate.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# import attention modules (we wrote these earlier)
from .attention import ConvolutionalShuffleAttention, MultiScaleShuffleAttention, DenseShuffleAttention

# small conv block helper
def conv_block(in_ch, out_ch, kernel=3, padding=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=True)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class AffineCouplingNet(nn.Module):
    """
    Small helper net used for Phi, rho, eta.
    Optionally uses an attention module injected at the middle.
    """
    def __init__(self, in_ch:int, out_ch:int, mid_ch:int=32, attn:Optional[nn.Module]=None):
        super().__init__()
        self.conv1 = conv_block(in_ch, mid_ch, kernel=3, padding=1)
        self.attn = attn
        # project back to out_ch
        self.conv2 = conv_block(mid_ch, out_ch, kernel=3, padding=1, act=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.attn is not None:
            x = x + self.attn(x)   # residual style fusion with attention output
        x = self.conv2(x)
        return x

class InvertibleBlock(nn.Module):
    """
    One invertible block (affine coupling style) that operates on two tensors:
    x1, x2 with same channel count.

    Args:
        channels: number of channels per input (e.g. 12)
        mid_ch: hidden channels inside Phi/rho/eta
        attn_groups: groups to pass to attention modules (if used)
        sigma_scale: scaling factor applied after sigmoid for the exponent (paper uses scaled sigmoid)
    """
    def __init__(self, channels:int=12, mid_ch:int=32, attn_groups:int=8, sigma_scale:float=1.0):
        super().__init__()
        self.channels = channels
        self.mid_ch = mid_ch
        self.sigma_scale = float(sigma_scale)

        # Choose attention modules for Phi, rho, eta
        # You can swap these classes if you prefer different configurations.
        self.phi_attn = ConvolutionalShuffleAttention(in_channels=mid_ch, hidden_channels=mid_ch, groups=attn_groups)
        self.rho_attn = MultiScaleShuffleAttention(in_channels=mid_ch, mid_channels=mid_ch//2, dilation_rates=[1,2,5], groups=attn_groups)
        self.eta_attn = DenseShuffleAttention(in_channels=mid_ch, growth=16, layers=3, groups=attn_groups)

        # Define Phi, rho, eta nets
        # Phi takes x2 -> outputs same shape as x1 (channels)
        self.Phi = AffineCouplingNet(in_ch=channels, out_ch=channels, mid_ch=mid_ch, attn=self.phi_attn)
        # rho maps y1 -> channels (used in exponent)
        self.Rho = AffineCouplingNet(in_ch=channels, out_ch=channels, mid_ch=mid_ch, attn=self.rho_attn)
        # eta maps y1 -> channels (additive term)
        self.Eta = AffineCouplingNet(in_ch=channels, out_ch=channels, mid_ch=mid_ch, attn=self.eta_attn)

        # small constant clamp for exp output to avoid overflow
        self.max_scale = 50.0  # clamp exponent to avoid inf

    def sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled sigmoid as in paper: sigma(x) = scale * sigmoid(x)
        We use sigma_scale hyperparam to adjust magnitude.
        """
        return torch.sigmoid(x) * self.sigma_scale

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward hide step: given x1 (cover-like) and x2 (secret-like) compute next pair y1,y2.
        Expects shapes: (B, C, H, W) for both, same C.
        """
        # Phi(x2)
        phi_out = self.Phi(x2)
        y1 = x1 + phi_out

        # compute rho and eta on y1
        rho_out = self.Rho(y1)
        eta_out = self.Eta(y1)

        # sigma(rho_out) scaled sigmoid, then exponentiate
        s = self.sigma(rho_out)
        # clamp s to avoid very large exponents (stability)
        s = torch.clamp(s, min=-self.max_scale, max=self.max_scale)
        # multiplicative term
        mult = torch.exp(s)

        y2 = eta_out + x2 * mult
        return y1, y2

    def inverse(self, y1: torch.Tensor, y2: torch.Tensor):
        """
        Inverse (recovery) step:
        Given y1, y2 compute original x1, x2 using algebraic inversion.
        """
        # compute rho(y1) and eta(y1)
        rho_out = self.Rho(y1)
        eta_out = self.Eta(y1)

        s = self.sigma(rho_out)
        s = torch.clamp(s, min=-self.max_scale, max=self.max_scale)
        mult = torch.exp(s)

        # x2 = (y2 - eta(y1)) * exp(-sigma(rho(y1)))
        inv_mult = torch.exp(-s)
        x2 = (y2 - eta_out) * inv_mult

        # x1 = y1 - Phi(x2)
        phi_out = self.Phi(x2)
        x1 = y1 - phi_out

        return x1, x2


# -----------------------
# Smoke test to validate forward <-> inverse correctness
# -----------------------
def _smoke_test():
    print("Running INN block smoke test...")
    B, C, H, W = 2, 12, 64, 64
    block = InvertibleBlock(channels=C, mid_ch=32, attn_groups=8, sigma_scale=1.0)
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)

    # forward
    y1, y2 = block(x1, x2)

    # inverse
    x1_rec, x2_rec = block.inverse(y1, y2)

    err1 = (x1 - x1_rec).abs().mean().item()
    err2 = (x2 - x2_rec).abs().mean().item()
    print(f"Reconstruction error x1: {err1:.6e}, x2: {err2:.6e}")
    # they should be near machine precision (or small float differences)
    assert err1 < 1e-6 and err2 < 1e-6, "Forward->Inverse did not perfectly reconstruct!"
    print("INN block smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
