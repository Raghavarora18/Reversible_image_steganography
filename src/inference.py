import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import math

from .config import DEVICE, STEGO_DIR, RECOVERED_DIR, TEST_DIR
from .model.risranet import RISRANet
from .losses import to_0_1, psnr, ssim

# helpers
def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return img

def pil_to_tensor(img):
    # returns tensor in [-1,1]
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])  # [0,1] -> [-1,1]
    return tf(img).unsqueeze(0)

def tensor_to_pil(tensor):
    # tensor expected in [-1,1], shape (1,3,H,W)
    t = tensor.detach().cpu().clamp(-1, 1)
    t01 = (t + 1.0) / 2.0
    t01 = t01.squeeze(0)
    tf_to_pil = T.ToPILImage()
    return tf_to_pil(t01)

def ensure_dirs():
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERED_DIR.mkdir(parents=True, exist_ok=True)

def save_pair(idx, cover_pil, stego_pil, recovered_pil, basename):
    # suffix filenames with idx to avoid overwriting
    s_name = f"{idx:04d}__{basename}"
    stego_path = STEGO_DIR / s_name
    rec_path = RECOVERED_DIR / s_name
    stego_pil.save(stego_path)
    recovered_pil.save(rec_path)
    return stego_path, rec_path

def compute_metrics(cover_t, stego_t, secret_t, rec_t):
    # inputs are tensors in [-1,1]
    cover01 = to_0_1(cover_t)
    stego01 = to_0_1(stego_t)
    secret01 = to_0_1(secret_t)
    rec01 = to_0_1(rec_t)

    psnr_cs = psnr(cover01, stego01)
    psnr_sr = psnr(secret01, rec01)
    ssim_cs = float(ssim(cover01, stego01).detach().cpu().item())
    ssim_sr = float(ssim(secret01, rec01).detach().cpu().item())
    return psnr_cs, psnr_sr, ssim_cs, ssim_sr

def load_checkpoint(model, path, device):
    ck = torch.load(path, map_location=device)
    # try both possible formats: dict with state_dict or raw state_dict
    if isinstance(ck, dict) and "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
    elif isinstance(ck, dict):
        model.load_state_dict(ck)
    else:
        model.load_state_dict(ck)
    return ck.get("epoch", None) if isinstance(ck, dict) else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover", type=str, default=None, help="path to cover image")
    parser.add_argument("--secret", type=str, default=None, help="path to secret image")
    parser.add_argument("--cover_dir", type=str, default=None, help="folder of cover images")
    parser.add_argument("--secret_dir", type=str, default=None, help="folder of secret images")
    parser.add_argument("--checkpoint", type=str, default=str(Path("checkpoints/best.pt")), help="path to checkpoint")
    parser.add_argument("--resize", type=int, default=None, help="resize short side to this (keeps square crops if used)")
    parser.add_argument("--max-samples", type=int, default=100, help="max samples to process in folder mode")
    parser.add_argument("--tb", action="store_true", help="log examples to tensorboard")
    args = parser.parse_args()

    device = DEVICE
    print("Device:", device)
    ensure_dirs()

    # build model
    model = RISRANet(in_channels=3).to(device)
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print("Loaded checkpoint:", args.checkpoint, "epoch:", epoch)

    writer = None
    if args.tb:
        tb_dir = Path("outputs") / "logs" / "inference_tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # single pair mode
    samples = []
    if args.cover and args.secret:
        samples.append((Path(args.cover), Path(args.secret)))
    elif args.cover_dir and args.secret_dir:
        cov_dir = Path(args.cover_dir)
        sec_dir = Path(args.secret_dir)
        cov_files = sorted([p for p in cov_dir.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        sec_files = sorted([p for p in sec_dir.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        # match by name if counts equal, else pair by index
        if len(cov_files) == len(sec_files):
            pairs = list(zip(cov_files, sec_files))
        else:
            # pair first N of each
            n = min(len(cov_files), len(sec_files), args.max_samples)
            pairs = [(cov_files[i], sec_files[i]) for i in range(n)]
        samples = pairs[: args.max_samples]
    else:
        # fallback: try TEST_DIR files as both cover & secret (use same image as secret)
        td = Path(TEST_DIR)
        if td.exists():
            files = sorted([p for p in td.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])[: args.max_samples]
            samples = [(f, f) for f in files]
        else:
            raise ValueError("No input provided and TEST_DIR does not exist or is empty.")

    results = []
    for idx, (cover_p, secret_p) in enumerate(samples, 1):
        cover_img = load_image(cover_p, size=args.resize) if args.resize else load_image(cover_p)
        secret_img = load_image(secret_p, size=args.resize) if args.resize else load_image(secret_p)

        cover_t = pil_to_tensor(cover_img).to(device)
        secret_t = pil_to_tensor(secret_img).to(device)

        with torch.no_grad():
            stego_t, g = model.hide(cover_t, secret_t)
            cover_rec_t, secret_rec_t = model.recover(stego_t, g=g)

        # save images
        stego_pil = tensor_to_pil(stego_t)
        rec_pil = tensor_to_pil(secret_rec_t)

        basename = f"{cover_p.stem}.png"
        s_path, r_path = save_pair(idx, cover_img, stego_pil, rec_pil, basename)

        # compute metrics
        psnr_cs, psnr_sr, ssim_cs, ssim_sr = compute_metrics(cover_t, stego_t, secret_t, secret_rec_t)
        results.append({
            "idx": idx,
            "cover": str(cover_p),
            "secret": str(secret_p),
            "stego": str(s_path),
            "recovered": str(r_path),
            "psnr_cs": psnr_cs,
            "psnr_sr": psnr_sr,
            "ssim_cs": ssim_cs,
            "ssim_sr": ssim_sr,
        })

        # tensorboard: log first few examples
        if writer and idx <= 8:
            writer.add_image(f"cover/{idx}", (to_0_1(cover_t).squeeze(0)).cpu(), 0)
            writer.add_image(f"stego/{idx}", (to_0_1(stego_t).squeeze(0)).cpu(), 0)
            writer.add_image(f"recovered/{idx}", (to_0_1(secret_rec_t).squeeze(0)).cpu(), 0)

        print(f"[{idx}] psnr_cs={psnr_cs:.2f}, psnr_sr={psnr_sr:.2f}, ssim_cs={ssim_cs:.4f}, ssim_sr={ssim_sr:.4f} -> stego: {s_path.name} recovered: {r_path.name}")

    # summary
    if len(results) > 0:
        avg_psnr_cs = sum(r["psnr_cs"] for r in results) / len(results)
        avg_psnr_sr = sum(r["psnr_sr"] for r in results) / len(results)
        avg_ssim_cs = sum(r["ssim_cs"] for r in results) / len(results)
        avg_ssim_sr = sum(r["ssim_sr"] for r in results) / len(results)
        print("=== SUMMARY ===")
        print(f"Samples: {len(results)}")
        print(f"AVG psnr_cs: {avg_psnr_cs:.2f}, AVG psnr_sr: {avg_psnr_sr:.2f}")
        print(f"AVG ssim_cs: {avg_ssim_cs:.4f}, AVG ssim_sr: {avg_ssim_sr:.4f}")

    if writer:
        writer.flush()
        writer.close()

if __name__ == "__main__":
    main()
