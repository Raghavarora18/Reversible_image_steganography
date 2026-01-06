
from typing import Tuple
import numpy as np
import math
import torch
import torch.nn.functional as F

# Optional pywt
try:
    import pywt
    _PYWT_AVAILABLE = True
except Exception as e:
    pywt = None
    _PYWT_AVAILABLE = False
    _PYWT_IMPORT_ERROR = e

# Constants
SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2

# -----------------------
# Numpy / pywt helpers (per-channel)
# -----------------------
def _ensure_pywt():
    if not _PYWT_AVAILABLE:
        raise ImportError(
            "pywt (PyWavelets) is required for non-'haar' wavelets. "
            "Install with: pip install pywavelets\n"
            f"Import error: {_PYWT_IMPORT_ERROR}"
        )

def dwt_forward_np(img: np.ndarray, wavelet: str = "haar") -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform 2D single-level DWT on each channel separately using pywt.
    Input: img: H x W x C (float in [0,1] or uint8)
    Returns: LL (H/2 x W/2 x C), (LH, HL, HH) each shape (H/2 x W/2 x C)
    """
    _ensure_pywt()
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    ll = []
    lh = []
    hl = []
    hh = []
    for c in range(C):
        coeffs2 = pywt.dwt2(img[:, :, c], wavelet)
        LL, (LH, HL, HH) = coeffs2
        ll.append(LL)
        lh.append(LH)
        hl.append(HL)
        hh.append(HH)
    # stack channels last
    LL = np.stack(ll, axis=-1)
    LH = np.stack(lh, axis=-1)
    HL = np.stack(hl, axis=-1)
    HH = np.stack(hh, axis=-1)
    return LL, (LH, HL, HH)

def dwt_inverse_np(LL: np.ndarray, highs: Tuple[np.ndarray, np.ndarray, np.ndarray], wavelet: str = "haar") -> np.ndarray:
    
    _ensure_pywt()
    C = LL.shape[-1]
    channels = []
    LH, HL, HH = highs
    for c in range(C):
        rec = pywt.idwt2((LL[..., c], (LH[..., c], HL[..., c], HH[..., c])), wavelet)
        channels.append(rec)
    img = np.stack(channels, axis=-1)
    return img

# -----------------------
# Pure-Torch Haar DWT / IWT (batch) - GPU-friendly
# -----------------------
def _pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
   
    b, c, h, w = x.shape
    pad_h = 0 if (h % 2 == 0) else 1
    pad_w = 0 if (w % 2 == 0) else 1
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom) for F.pad
    x_p = F.pad(x, pad=pad, mode='reflect')
    return x_p, (pad_h, pad_w)

def _remove_padding(x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad
    if pad_h == 0 and pad_w == 0:
        return x
    b, c, h, w = x.shape
    return x[:, :, :h - pad_h, :w - pad_w]

def dwt_forward_torch_haar(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Pure-Torch single-level Haar DWT (separable).
    Input: x (B, C, H, W)
    Output:
      LL, (LH, HL, HH) each (B, C, H//2, W//2)
    """
    assert x.ndim == 4
    device = x.device
    dtype = x.dtype

    # pad to even dims if needed
    x_p, pad = _pad_to_even(x)

    # row transform (pair adjacent rows)
    x_row0 = x_p[:, :, 0::2, :]
    x_row1 = x_p[:, :, 1::2, :]

    row_low = (x_row0 + x_row1) * INV_SQRT2
    row_high = (x_row0 - x_row1) * INV_SQRT2

    # column transform on row_low -> LL, LH
    row_low_col0 = row_low[:, :, :, 0::2]
    row_low_col1 = row_low[:, :, :, 1::2]
    LL = (row_low_col0 + row_low_col1) * INV_SQRT2
    LH = (row_low_col0 - row_low_col1) * INV_SQRT2

    # column transform on row_high -> HL, HH
    row_high_col0 = row_high[:, :, :, 0::2]
    row_high_col1 = row_high[:, :, :, 1::2]
    HL = (row_high_col0 + row_high_col1) * INV_SQRT2
    HH = (row_high_col0 - row_high_col1) * INV_SQRT2

    return LL.to(device=device, dtype=dtype), (LH.to(device=device, dtype=dtype),
                                               HL.to(device=device, dtype=dtype),
                                               HH.to(device=device, dtype=dtype))

def dwt_inverse_torch_haar(LL: torch.Tensor, highs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    
    LH, HL, HH = highs
    device = LL.device
    dtype = LL.dtype

    # reconstruct columns for row_low and row_high
    row_low_col0 = (LL + LH) * INV_SQRT2
    row_low_col1 = (LL - LH) * INV_SQRT2

    row_high_col0 = (HL + HH) * INV_SQRT2
    row_high_col1 = (HL - HH) * INV_SQRT2

    B, C, H2, W2 = LL.shape
    Wp = W2 * 2

    row_low = torch.empty((B, C, H2, Wp), device=device, dtype=dtype)
    row_high = torch.empty((B, C, H2, Wp), device=device, dtype=dtype)

    row_low[:, :, :, 0::2] = row_low_col0
    row_low[:, :, :, 1::2] = row_low_col1

    row_high[:, :, :, 0::2] = row_high_col0
    row_high[:, :, :, 1::2] = row_high_col1

    # invert row transform
    x_row0 = (row_low + row_high) * INV_SQRT2
    x_row1 = (row_low - row_high) * INV_SQRT2

    H = H2 * 2
    x_rec = torch.empty((B, C, H, Wp), device=device, dtype=dtype)
    x_rec[:, :, 0::2, :] = x_row0
    x_rec[:, :, 1::2, :] = x_row1

    return x_rec

# -----------------------
# Wrapper functions used by RISRANet (preferred API)
# -----------------------
def dwt_forward_torch(x: torch.Tensor, wavelet: str = "haar"):
    """
    Wrapper: returns LL and (LH,HL,HH).
    - For wavelet='haar': uses pure-torch implementation.
    - For others: uses pywt-based numpy functions if pywt is available.
      If pywt is not available, raises NotImplementedError.
    """
    if wavelet == "haar":
        return dwt_forward_torch_haar(x)
    # fallback to numpy/pywt if available
    if _PYWT_AVAILABLE:
        # convert batch to numpy and call dwt_forward_np per sample
        b, c, h, w = x.shape
        x_np = x.detach().cpu().numpy()
        LLs, LHs, HLs, HHs = [], [], [], []
        for i in range(b):
            img = np.transpose(x_np[i], (1, 2, 0))  # H,W,C
            LL, (LH, HL, HH) = dwt_forward_np(img, wavelet=wavelet)
            LLs.append(np.transpose(LL, (2, 0, 1)))
            LHs.append(np.transpose(LH, (2, 0, 1)))
            HLs.append(np.transpose(HL, (2, 0, 1)))
            HHs.append(np.transpose(HH, (2, 0, 1)))
        LL_t = torch.from_numpy(np.stack(LLs, axis=0)).to(x.device).float()
        LH_t = torch.from_numpy(np.stack(LHs, axis=0)).to(x.device).float()
        HL_t = torch.from_numpy(np.stack(HLs, axis=0)).to(x.device).float()
        HH_t = torch.from_numpy(np.stack(HHs, axis=0)).to(x.device).float()
        return LL_t, (LH_t, HL_t, HH_t)
    raise NotImplementedError("Only 'haar' is implemented by the pure-torch DWT. Install pywavelets for other wavelets.")

def dwt_inverse_torch(LL: torch.Tensor, highs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], wavelet: str = "haar"):
    """
    Wrapper for inverse DWT.
    - For 'haar' uses pure-torch inverse.
    - For others, uses pywt-based numpy helper if available.
    """
    if wavelet == "haar":
        # inverse returns padded result; caller must handle cropping if needed
        return dwt_inverse_torch_haar(LL, highs)
    if _PYWT_AVAILABLE:
        LL_np = LL.detach().cpu().numpy()
        LH_np = highs[0].detach().cpu().numpy()
        HL_np = highs[1].detach().cpu().numpy()
        HH_np = highs[2].detach().cpu().numpy()
        b = LL_np.shape[0]
        recs = []
        for i in range(b):
            LL_hw = np.transpose(LL_np[i], (1, 2, 0))   # H2,W2,C
            LH_hw = np.transpose(LH_np[i], (1, 2, 0))
            HL_hw = np.transpose(HL_np[i], (1, 2, 0))
            HH_hw = np.transpose(HH_np[i], (1, 2, 0))
            rec = dwt_inverse_np(LL_hw, (LH_hw, HL_hw, HH_hw), wavelet=wavelet)
            recs.append(np.transpose(rec, (2, 0, 1)))
        rec_t = torch.from_numpy(np.stack(recs, axis=0)).to(LL.device).float()
        return rec_t
    raise NotImplementedError("Only 'haar' is implemented by the pure-torch IWT. Install pywavelets for other wavelets.")

# -----------------------
# Smoke tests (run when executed directly)
# -----------------------
if __name__ == "__main__":
    print("Running DWT/IWT smoke tests...")
    # basic torch/haar roundtrip
    t = torch.rand(1, 3, 129, 127)  # intentionally odd dims to test padding
    LL, highs = dwt_forward_torch(t, wavelet="haar")
    rec = dwt_inverse_torch(LL, highs, wavelet="haar")
    # remove padding (we know original dims)
    rec_cropped = rec[:, :, :t.shape[2], :t.shape[3]]
    err = (t - rec_cropped).abs().mean().item()
    print(f"Haar torch roundtrip mean abs error: {err:.6e}")

    # if pywt present, run small numpy/pywt roundtrip test
    if _PYWT_AVAILABLE:
        print("pywt available — testing numpy/pywt helpers...")
        import numpy as _np
        img = (_np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        LLn, highs_n = dwt_forward_np(img / 255.0, wavelet="haar")
        rec_n = dwt_inverse_np(LLn, highs_n, wavelet="haar")
        err_np = np.abs(rec_n - (img / 255.0)).mean()
        print(f"pywt numpy roundtrip mean abs error: {err_np:.6e}")
    else:
        print("pywt not installed — numpy/pywt tests skipped.")
