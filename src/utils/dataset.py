# src/utils/dataset.py
"""
StegoDataset: a PyTorch Dataset that yields (cover, secret) image pairs.

Behavior:
- Expects two folders: cover_dir and secret_dir (files matched by filename order).
- On training: random crop + optional transforms.
- On validation/test: center crop / resize.
- Returns tensors in range [-1, 1], shape: (C, H, W).
- Includes helper get_dataloaders(...) to get train/val dataloaders.

You can test this dataset without a real dataset by pointing cover_dir/secret_dir
to small folders with a few images, or by using the `test_with_dummy()` helper below.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import random
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from ..config import TRAIN_CROP_SIZE, TEST_CROP_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE
from .image_io import load_image, pil_to_tensor, normalize_tensor, center_crop, resize_image

# -----------------------
# Utility: collect image files
# -----------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return files

# -----------------------
# StegoDataset
# -----------------------
class StegoDataset(Dataset):
    def __init__(
        self,
        cover_dir: str,
        secret_dir: str,
        crop_size: int = TRAIN_CROP_SIZE,
        mode: str = "train",
        max_pairs: Optional[int] = None,
    ):
        """
        cover_dir, secret_dir: paths to folders containing images.
        crop_size: int, crop/resize size (square).
        mode: 'train' or 'val' or 'test'
        max_pairs: optional, limit dataset size for quick debugging.
        """
        self.cover_dir = Path(cover_dir)
        self.secret_dir = Path(secret_dir)
        self.crop_size = int(crop_size)
        self.mode = mode.lower()
        assert self.mode in ("train", "val", "test")
        self.covers = _list_images(self.cover_dir)
        self.secrets = _list_images(self.secret_dir)

        # pair images by order; if lengths differ, we cycle the shorter list
        if len(self.covers) == 0 or len(self.secrets) == 0:
            raise RuntimeError(f"Empty cover or secret folder: {cover_dir}, {secret_dir}")

        if len(self.covers) >= len(self.secrets):
            self.pairs = [(self.covers[i], self.secrets[i % len(self.secrets)]) for i in range(len(self.covers))]
        else:
            self.pairs = [(self.covers[i % len(self.covers)], self.secrets[i]) for i in range(len(self.secrets))]

        if max_pairs is not None:
            self.pairs = self.pairs[:int(max_pairs)]

    def __len__(self):
        return len(self.pairs)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        pil = load_image(str(path))  # PIL RGB
        if self.mode == "train":
            # random resize/crop: if smaller than crop_size, upscale then random crop
            if pil.width < self.crop_size or pil.height < self.crop_size:
                pil = pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            # random crop
            i = random.randint(0, pil.height - self.crop_size) if pil.height > self.crop_size else 0
            j = random.randint(0, pil.width - self.crop_size) if pil.width > self.crop_size else 0
            pil = pil.crop((j, i, j + self.crop_size, i + self.crop_size))
        else:
            # val/test: center crop (or resize then center crop)
            if pil.width < self.crop_size or pil.height < self.crop_size:
                pil = pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            pil = center_crop(pil, (self.crop_size, self.crop_size))

        t = pil_to_tensor(pil)            # [0,1] C,H,W
        t = normalize_tensor(t)          # [-1,1]
        return t

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        cover_path, secret_path = self.pairs[idx]
        cover_t = self._load_and_preprocess(cover_path)
        secret_t = self._load_and_preprocess(secret_path)
        return cover_t, secret_t

# -----------------------
# Dataloader helper
# -----------------------
def get_dataloaders(
    cover_dir: str,
    secret_dir: str,
    batch_size: int = BATCH_SIZE,
    crop_size: int = TRAIN_CROP_SIZE,
    val_split: float = 0.05,
    num_workers: int = NUM_WORKERS,
    max_pairs: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Returns: train_loader, val_loader (val_loader may be None if dataset too small)
    Simple split: first (1-val_split) portion used for train, remaining for val.
    """
    # create full dataset (mode='train' used for transformations)
    full = StegoDataset(cover_dir, secret_dir, crop_size=crop_size, mode="train", max_pairs=max_pairs)
    n = len(full)
    if n == 0:
        raise RuntimeError("No image pairs found.")

    # compute split sizes
    val_n = max(1, int(n * val_split)) if n > 1 else 0
    train_n = n - val_n

    # indices
    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx = indices[train_n:] if val_n > 0 else []

    # Subset samplers / datasets:
    from torch.utils.data import Subset
    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx) if val_n > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds is not None else None

    return train_loader, val_loader

# -----------------------
# Quick smoke-test without dataset (create dummy images in memory)
# -----------------------
def test_with_dummy(num=8, crop_size=256):
    """
    Quick unit test that creates in-memory images (random) and checks dataloader shapes.
    Does NOT require saving image files.
    """
    import numpy as np
    from io import BytesIO

    # create temporary folders and save a few images
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    cov_dir = Path(tmp.name) / "cover"
    sec_dir = Path(tmp.name) / "secret"
    cov_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)

    for i in range(max(4, num)):
        arr_c = (np.random.rand(crop_size, crop_size, 3) * 255).astype("uint8")
        arr_s = (np.random.rand(crop_size, crop_size, 3) * 255).astype("uint8")
        Image.fromarray(arr_c).save(cov_dir / f"c_{i:03d}.png")
        Image.fromarray(arr_s).save(sec_dir / f"s_{i:03d}.png")

    # get loaders
    train_loader, val_loader = get_dataloaders(str(cov_dir), str(sec_dir), batch_size=min(4, num), crop_size=crop_size, val_split=0.25, num_workers=0)
    for batch in train_loader:
        cover_batch, secret_batch = batch  # each: (B, C, H, W)
        print("cover:", cover_batch.shape, "secret:", secret_batch.shape)
        break

    tmp.cleanup()

# -----------------------
# Run quick test if executed directly
# -----------------------
if __name__ == "__main__":
    print("Running dataset quick test (creates temporary dummy images)...")
    test_with_dummy()
