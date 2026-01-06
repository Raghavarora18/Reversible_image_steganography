# src/model/risranet.py
"""
RISRANet: stacks multiple invertible blocks and handles DWT/IWT preprocessing.

- hide(cover, secret) -> returns stego_tensor, lost_info_tensor (r)
  * inputs: cover, secret = tensors in range [-1,1], shape (B, C, H, W)
  * outputs: stego in [-1,1], and r (DWT-domain tensor)

- recover(stego, g=None, seed=None) -> returns cover_rec, secret_rec
  * if g is provided it will be used as the "lost info" (exact recovery possible)
  * if g is None, random Gaussian noise is used (practical recovery)

This module depends on src.model.inn_block and src.utils.dwt_iwt
"""

from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_INVERTIBLE_BLOCKS, DEVICE, WAVELET
from .inn_block import InvertibleBlock
from ..utils.dwt_iwt import dwt_forward_torch, dwt_inverse_torch


class RISRANet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = NUM_INVERTIBLE_BLOCKS,
        mid_ch: int = 32,
        attn_groups: int = 8,
        sigma_scale: float = 1.0,
        wavelet: str = WAVELET,
        device: Optional[str] = None,
    ):
        """
        in_channels: number of image color channels (usually 3)
        num_blocks: number of invertible blocks to stack
        mid_ch: mid channels inside Phi/rho/eta
        attn_groups: groups passed to attention modules
        sigma_scale: scaling for sigmoid before exponent in block
        wavelet: name passed to DWT/IWT helpers
        """
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = wavelet
        self.device = device if device is not None else DEVICE

        # After single-level DWT, each channel produces 4 subbands (LL,LH,HL,HH)
        self.coeff_channels = in_channels * 4

        # Create a list of invertible blocks
        self.blocks = nn.ModuleList([
            InvertibleBlock(channels=self.coeff_channels, mid_ch=mid_ch, attn_groups=attn_groups, sigma_scale=sigma_scale)
            for _ in range(num_blocks)
        ])

    # -----------------------
    # Helpers: packing/unpacking DWT subbands
    # -----------------------
    @staticmethod
    def pack_coeffs(LL, highs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Given LL (B,C,H2,W2) and highs tuple (LH,HL,HH) all same shapes,
        return concatenated tensor (B, 4*C, H2, W2) in order [LL, LH, HL, HH].
        """
        LH, HL, HH = highs
        return torch.cat([LL, LH, HL, HH], dim=1)

    @staticmethod
    def unpack_coeffs(x: torch.Tensor, c: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Split concatenated coeffs (B,4C,H2,W2) into LL and (LH,HL,HH).
        c = original number of channels
        """
        LL = x[:, 0:c, :, :]
        LH = x[:, c:2*c, :, :]
        HL = x[:, 2*c:3*c, :, :]
        HH = x[:, 3*c:4*c, :, :]
        return LL, (LH, HL, HH)

    # -----------------------
    # Core API: hide & recover
    # -----------------------
    def hide(self, cover: torch.Tensor, secret: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hide secret inside cover.
        Inputs:
            cover, secret: tensors in [-1,1], shape (B, C, H, W)
        Returns:
            stego: tensor in [-1,1] shape (B, C, H, W)
            r: the 'lost info' tensor in DWT-domain (B, 4C, H2, W2) — keep if you want perfect recovery
        """
        assert cover.shape == secret.shape, "cover and secret must have same shape"
        # convert to [0,1] for DWT
        cover_01 = (cover + 1.0) / 2.0
        secret_01 = (secret + 1.0) / 2.0

        # apply DWT -> get LL and highs per image; outputs are shapes (B,C,H2,W2)
        LL_c, highs_c = dwt_forward_torch(cover_01, wavelet=self.wavelet)
        LL_s, highs_s = dwt_forward_torch(secret_01, wavelet=self.wavelet)

        # pack coefficients: make xcover and xsecret in DWT-domain with 4*C channels
        xcover = self.pack_coeffs(LL_c, highs_c)  # (B,4C,H2,W2)
        xsecret = self.pack_coeffs(LL_s, highs_s)

        # pass through stacked invertible blocks (forward direction)
        x1, x2 = xcover, xsecret
        for blk in self.blocks:
            x1, x2 = blk(x1, x2)  # each returns y1,y2

        # final x1 is xstego (DWT domain), x2 is r (lost info)
        xstego = x1
        r = x2

        # reconstruct stego image via IWT (split xstego back to LL,LH,HL,HH)
        LL_stego, highs_stego = self.unpack_coeffs(xstego, self.in_channels)
        stego_01 = dwt_inverse_torch(LL_stego, highs_stego, wavelet=self.wavelet)  # [0,1]
        stego = stego_01 * 2.0 - 1.0  # back to [-1,1]

        return stego.to(self.device), r.to(self.device)

    def recover(self, stego: torch.Tensor, g: Optional[torch.Tensor] = None, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover cover and secret from a stego image.
        Inputs:
            stego: tensor in [-1,1], shape (B,C,H,W)
            g: optional tensor (B, 4C, H2,W2) to be used as lost-info. If provided => exact recovery.
               If None => g is sampled as Gaussian noise with same shape.
            seed: optional seed to make g deterministic
        Returns:
            cover_rec, secret_rec: tensors in [-1,1], shape (B,C,H,W)
        """
        # convert stego to [0,1] for DWT
        stego_01 = (stego + 1.0) / 2.0

        # DWT on stego to get xstego coefficients
        LL_stego, highs_stego = dwt_forward_torch(stego_01, wavelet=self.wavelet)
        xstego = self.pack_coeffs(LL_stego, highs_stego)  # (B,4C,H2,W2)

        # prepare g noise if not given. g must have same shape as xsecret/r (B,4C,H2,W2)
        if g is None:
            if seed is not None:
                torch.manual_seed(seed)
            g = torch.randn_like(xstego, device=xstego.device)

        # run inverse through blocks in reverse order
        y1, y2 = xstego, g
        for blk in reversed(self.blocks):
            y1, y2 = blk.inverse(y1, y2)

        # y1 is xcover_rev (DWT-domain), y2 is xsecret_rev (DWT-domain)
        xcover_rev = y1
        xsecret_rev = y2

        # reconstruct images via IWT
        LL_c_rev, highs_c_rev = self.unpack_coeffs(xcover_rev, self.in_channels)
        cover_rec_01 = dwt_inverse_torch(LL_c_rev, highs_c_rev, wavelet=self.wavelet)
        cover_rec = cover_rec_01 * 2.0 - 1.0

        LL_s_rev, highs_s_rev = self.unpack_coeffs(xsecret_rev, self.in_channels)
        secret_rec_01 = dwt_inverse_torch(LL_s_rev, highs_s_rev, wavelet=self.wavelet)
        secret_rec = secret_rec_01 * 2.0 - 1.0

        return cover_rec.to(self.device), secret_rec.to(self.device)

    # -----------------------
    # Checkpoint helpers
    # -----------------------
    def save_checkpoint(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
        }, str(p))

    def load_checkpoint(self, path: str, map_location: Optional[str] = None):
        map_loc = map_location if map_location is not None else self.device
        ckpt = torch.load(str(path), map_location=map_loc)
        self.load_state_dict(ckpt["state_dict"])

    # -----------------------
    # Smoke test (forward->inverse using exact r)
    # -----------------------
def _smoke_test():
    """
    - Create a random cover & secret (B=1), run hide() to obtain stego+r,
    - Run recover(stego, g=r) and verify recovered images match originals (small error).
    """
    import torch
    from ..config import DEVICE
    print("Running RISRANet smoke test...")
    device = DEVICE
    net = RISRANet(in_channels=3, num_blocks=4, mid_ch=32, attn_groups=8, device=device).to(device)

    B, C, H, W = 1, 3, 128, 128
    cover = torch.rand(B, C, H, W, device=device) * 2.0 - 1.0   # [-1,1]
    secret = torch.rand(B, C, H, W, device=device) * 2.0 - 1.0  # [-1,1]

    stego, r = net.hide(cover, secret)
    cover_rec, secret_rec = net.recover(stego, g=r)

    err_cover = (cover - cover_rec).abs().mean().item()
    err_secret = (secret - secret_rec).abs().mean().item()
    print(f"Reconstruction errors — cover: {err_cover:.6e}, secret: {err_secret:.6e}")
    assert err_cover < 1e-6 and err_secret < 1e-6, "Exact recover failed!"
    print("RISRANet smoke test passed.")

# allow running as module
if __name__ == "__main__":
    _smoke_test()
