
from pathlib import Path
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = ROOT / "outputs"
STEGO_DIR = OUTPUT_DIR / "stego"
RECOVERED_DIR = OUTPUT_DIR / "recovered"
LOGS_DIR = OUTPUT_DIR / "logs"

CHECKPOINTS = ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINTS / "best.pt"
LATEST_CHECKPOINT = CHECKPOINTS / "latest.pt"

for p in (DATA_DIR, OUTPUT_DIR, STEGO_DIR, RECOVERED_DIR, LOGS_DIR, CHECKPOINTS):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# image sizes
TRAIN_CROP_SIZE = 256
TEST_CROP_SIZE = 256
WAVELET = "haar"

# model defaults (keep as-is)
NUM_INVERTIBLE_BLOCKS = 8
MID_CH = 64

# training defaults (fine-tune friendly)
BATCH_SIZE = 12
NUM_WORKERS = 12
EPOCHS = 30
LR = 3e-6
SEED = 42

# *** Loss weights (refinement default) ***
# these are chosen to favor cover fidelity for final push
LAMBDA_H = 10.0     # hide / cover fidelity (increase)
LAMBDA_R = 0.2      # recovery (reduced; already good)
LAMBDA_F = 1.0      # low-frequency (wavelet) penalty
LAMBDA_S = 1.0      # SSIM structural
LAMBDA_P = 0.0      # perceptual (default off for final refinement)

# fine-tune params
LR_FINETUNE = 1e-6
FINETUNE_EPOCHS = 30

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# opt-in: call set_seed on import if you prefer deterministic runs
set_seed(SEED)
