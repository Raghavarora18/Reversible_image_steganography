from typing import Tuple
import torch
import torch.nn.functional as F
import math

from .config import LAMBDA_H, LAMBDA_R, LAMBDA_F, LAMBDA_S, LAMBDA_P, WAVELET

# DWT helper (must exist in your repo)
from .utils.dwt_iwt import dwt_forward_torch

# optional lpips
try:
    import lpips
    _LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    _LPIPS_AVAILABLE = False

# torchvision fallback
try:
    from torchvision import models
    _TV_AVAILABLE = True
except Exception:
    models = None
    _TV_AVAILABLE = False

# --------------------------
# Basic helpers
# --------------------------
def to_0_1(img: torch.Tensor) -> torch.Tensor:
    return (img + 1.0) / 2.0

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="mean")

def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
    mse_val = F.mse_loss(a, b, reduction="mean").item()
    if mse_val == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse_val)

# --------------------------
# SSIM (simple differentiable)
# --------------------------
def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = g.unsqueeze(1) @ g.unsqueeze(0)
    return w

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, K1: float = 0.01, K2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    assert img1.shape == img2.shape
    _, C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype
    window = _gaussian_window(window_size, sigma=1.5, device=device, dtype=dtype)
    window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, groups=C, padding=pad)
    mu2 = F.conv2d(img2, window, groups=C, padding=pad)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=C, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=C, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=C, padding=pad) - mu1_mu2

    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1,2,3]).mean()

def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(to_0_1(img1), to_0_1(img2))

# --------------------------
# Perceptual loss (LPIPS or VGG)
# --------------------------
_perceptual_model = None
def _init_perceptual(device):
    global _perceptual_model
    if _perceptual_model is not None:
        return _perceptual_model
    if _LPIPS_AVAILABLE:
        _perceptual_model = lpips.LPIPS(net='vgg').to(device)
        _perceptual_model.eval()
    elif _TV_AVAILABLE:
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        _perceptual_model = vgg
    else:
        _perceptual_model = None
    return _perceptual_model

def perceptual_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inputs a,b expected in [-1,1].
    Returns scalar tensor.
    """
    device = a.device
    model = _init_perceptual(device)
    if model is None:
        return mse(a, b)

    if _LPIPS_AVAILABLE:
        # LPIPS expects [-1,1]
        with torch.no_grad():
            return model(a, b).mean()
    else:
        # VGG fallback: compare early feature maps
        def to_vgg_input(x):
            x01 = (x + 1.0) / 2.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
            return (x01 - mean) / std

        a_v = to_vgg_input(a)
        b_v = to_vgg_input(b)
        vgg = model
        feats_a = []
        feats_b = []
        x_a = a_v
        x_b = b_v
        layer_indices = [3, 8, 15]  # relu1_2, relu2_2, relu3_3
        for i, layer in enumerate(vgg.children()):
            x_a = layer(x_a)
            x_b = layer(x_b)
            if i in layer_indices:
                feats_a.append(x_a)
                feats_b.append(x_b)
            if i >= max(layer_indices):
                break
        loss = 0.0
        for fa, fb in zip(feats_a, feats_b):
            loss = loss + F.mse_loss(fa, fb, reduction='mean')
        return loss

# --------------------------
# Loss components
# --------------------------
def hiding_loss(stego: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    return mse(stego, cover)

def recovery_loss(secret_rev: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
    return mse(secret_rev, secret)

def lowfreq_loss(stego: torch.Tensor, cover: torch.Tensor, wavelet: str = WAVELET) -> torch.Tensor:
    a = to_0_1(stego)
    b = to_0_1(cover)
    LL_a, _ = dwt_forward_torch(a, wavelet=wavelet)
    LL_b, _ = dwt_forward_torch(b, wavelet=wavelet)
    return mse(LL_a, LL_b)

def structural_similarity_loss(cover: torch.Tensor, stego: torch.Tensor, secret: torch.Tensor, secret_rev: torch.Tensor) -> torch.Tensor:
    loss1 = 1.0 - ssim(to_0_1(stego), to_0_1(cover))
    loss2 = 1.0 - ssim(to_0_1(secret_rev), to_0_1(secret))
    return loss1 + loss2

# --------------------------
# Total loss wrapper
# --------------------------
def total_loss(cover, stego, secret, secret_rev, wavelet=WAVELET):
    Lhid = hiding_loss(stego, cover)
    Lrec = recovery_loss(secret_rev, secret)
    Lfreq = lowfreq_loss(stego, cover, wavelet=wavelet)
    Lssim = structural_similarity_loss(cover, stego, secret, secret_rev)

    # ↓  THIS IS THE NEW PART ↓
    if torch.rand(1).item() > 0.5:
        Lper = perceptual_loss(stego, cover)
    else:
        Lper = torch.tensor(0.0, device=cover.device)

    total = (
        LAMBDA_H * Lhid
        + LAMBDA_R * Lrec
        + LAMBDA_F * Lfreq
        + LAMBDA_S * Lssim
        + LAMBDA_P * Lper
    )

    diagnostics = {
        "Lhid": float(Lhid.detach().cpu().item()),
        "Lrec": float(Lrec.detach().cpu().item()),
        "Lfreq": float(Lfreq.detach().cpu().item()),
        "Lssim": float(Lssim.detach().cpu().item()),
        "Lper": float(Lper.detach().cpu().item()),
        "Ltotal": float(total.detach().cpu().item()),
    }

    return total, diagnostics

# --------------------------
# Smoke test
# --------------------------
if __name__ == "__main__":
    print("Running losses smoke test...")
    B, C, H, W = 2, 3, 128, 128
    cover = torch.randn(B, C, H, W).clamp(-1, 1)
    secret = torch.randn(B, C, H, W).clamp(-1, 1)
    stego = (cover + 0.01 * torch.randn_like(cover)).clamp(-1, 1)
    secret_rev = (secret + 0.01 * torch.randn_like(secret)).clamp(-1, 1)
    tot, d = total_loss(cover, stego, secret, secret_rev)
    print("Diagnostics:", d)
    print("Total:", tot.item())
