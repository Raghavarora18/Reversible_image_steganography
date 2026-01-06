# src/utils/image_io.py
"""
Utility functions for loading, saving, and converting images.
Used across dataset, model, and UI layers.
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# ---------------------------
# Loading & Saving Images
# ---------------------------

def load_image(path):
    """
    Loads an image from disk and returns a PIL Image in RGB.
    """
    img = Image.open(path).convert("RGB")
    return img


def save_image(img, path):
    """
    Saves a tensor or numpy image to disk.
    img can be:
        - torch.Tensor CxHxW (float 0..1)
        - numpy array HxWxC (float or uint8)
    """
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)

    # Convert float to uint8 if needed
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(path)


# ---------------------------
# Conversion Helpers
# ---------------------------

def pil_to_tensor(img):
    """
    Converts PIL → torch tensor in shape CxHxW normalized to [0,1].
    """
    return TF.to_tensor(img)  # already returns float [0,1]


def tensor_to_numpy(t: torch.Tensor):
    """
    Convert tensor CxHxW or BxCxHxW → numpy HxWxC.
    """
    if t.ndim == 4:  # take first image from batch
        t = t[0]

    # C,H,W → H,W,C
    img = t.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


# ---------------------------
# Normalization Helpers
# ---------------------------

def normalize_tensor(img: torch.Tensor):
    """
    Normalizes tensor from [0,1] to [-1,1].
    """
    return img * 2 - 1


def denormalize_tensor(img: torch.Tensor):
    """
    Converts [-1,1] tensor back to [0,1].
    """
    return (img + 1) / 2


# ---------------------------
# Image Resize & Crop
# ---------------------------

def center_crop(pil_img, size):
    """
    Center crop for PIL images.
    """
    return TF.center_crop(pil_img, size)


def resize_image(pil_img, size):
    """
    Resize image (square or tuple).
    """
    return pil_img.resize((size, size), Image.BILINEAR)


# ---------------------------
# Batch Utilities
# ---------------------------

def prepare_image_for_model(pil_img, size=256):
    """
    Loads → resize → tensor → normalize.
    Used for inference (hide/recover.py)
    """
    img = resize_image(pil_img, size)
    t = pil_to_tensor(img)  # [0,1]
    t = normalize_tensor(t)  # [-1,1]
    return t.unsqueeze(0)  # add batch dim
