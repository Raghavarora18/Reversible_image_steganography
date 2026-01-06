import argparse
import time
import csv
from pathlib import Path
import os

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils as vutils

from .config import (
    DEVICE, BATCH_SIZE, LR, EPOCHS, NUM_WORKERS,
    BEST_CHECKPOINT, LATEST_CHECKPOINT, LOGS_DIR, SEED,
    TRAIN_CROP_SIZE, WAVELET, STEGO_DIR, RECOVERED_DIR
)
from .utils.dataset import get_dataloaders
from .model.risranet import RISRANet
from .losses import total_loss, to_0_1, ssim, psnr
import src.config as cfg_mod


def set_seed_local(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, path):
    ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer is not None else None}
    torch.save(ckpt, path)

def per_epoch_checkpoint(model, optimizer, epoch):
    path = Path("checkpoints") / f"ckpt_epoch_{epoch}.pt"
    save_checkpoint(model, optimizer, epoch, str(path))

def load_checkpoint(model, optimizer, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
        except Exception:
            # fallback: try common keys
            model.load_state_dict(ckpt.get("model_state", ckpt))
    else:
        model.load_state_dict(ckpt)
    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    return ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0

def validate(model, val_loader, device, wavelet, use_amp=False):
    model.eval()
    total_items = 0
    metrics = {"val_loss": 0.0, "psnr_cover_stego": 0.0, "psnr_secret_rec": 0.0, "ssim_cover_stego": 0.0, "ssim_secret_rec": 0.0}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Valid", leave=False):
            cover, secret = batch
            cover = cover.to(device); secret = secret.to(device)
            with autocast(enabled=use_amp):
                stego, r = model.hide(cover, secret)
                cover_rec, secret_rec = model.recover(stego, g=r)
                loss_tensor, _diag = total_loss(cover, stego, secret, secret_rec, wavelet=wavelet)
                batch_loss = float(loss_tensor.detach().cpu().item())
            bs = cover.shape[0]
            total_items += bs
            metrics["val_loss"] += batch_loss * bs
            cover_01 = to_0_1(cover); stego_01 = to_0_1(stego)
            secret_01 = to_0_1(secret); secret_rec_01 = to_0_1(secret_rec)
            metrics["psnr_cover_stego"] += psnr(cover_01, stego_01) * bs
            metrics["psnr_secret_rec"] += psnr(secret_01, secret_rec_01) * bs
            metrics["ssim_cover_stego"] += float(ssim(cover_01, stego_01).detach().cpu().item()) * bs
            metrics["ssim_secret_rec"] += float(ssim(secret_01, secret_rec_01).detach().cpu().item()) * bs
    if total_items > 0:
        for k in metrics: metrics[k] /= total_items
    model.train()
    return metrics

def train_epoch(model, train_loader, optimizer, device, scaler, use_amp, wavelet, epoch_idx, total_epochs, log_interval=50):
    model.train()
    running_loss = 0.0; items = 0
    iters_per_epoch = len(train_loader)
    start_time = time.time(); iter_count = 0
    pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Train E{epoch_idx}", leave=False)
    for i, batch in pbar:
        cover, secret = batch; cover = cover.to(device); secret = secret.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            stego, r = model.hide(cover, secret)
            cover_rec, secret_rec = model.recover(stego, g=r)
            loss_tensor, diag = total_loss(cover, stego, secret, secret_rec, wavelet=wavelet)
        if use_amp:
            scaler.scale(loss_tensor).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss_tensor.backward(); optimizer.step()
        bs = cover.shape[0]
        running_loss += float(loss_tensor.detach().cpu().item()) * bs
        items += bs; iter_count += 1
        elapsed = time.time() - start_time
        avg_sec_per_iter = elapsed / iter_count if iter_count > 0 else 0.0
        remaining_iters_epoch = max(0, iters_per_epoch - (i + 1))
        eta_epoch_sec = remaining_iters_epoch * avg_sec_per_iter
        remaining_epochs = max(0, total_epochs - epoch_idx)
        eta_full_sec = eta_epoch_sec + remaining_epochs * (iters_per_epoch * avg_sec_per_iter)
        if (i + 1) % log_interval == 0 or (i + 1) == iters_per_epoch:
            pbar.set_postfix({"loss": f"{(running_loss/items):.3e}", "epoch_eta": f"{int(eta_epoch_sec//60)}m{int(eta_epoch_sec%60)}s", "full_eta": f"{int(eta_full_sec//3600)}h{int((eta_full_sec%3600)//60)}m"})
    return running_loss / items if items > 0 else 0.0

def save_sample_images(model, device, val_loader, epoch, n_samples=4):
    model.eval()
    saved = 0
    for batch in val_loader:
        cover, secret = batch
        cover = cover.to(device); secret = secret.to(device)
        with torch.no_grad():
            stego, r = model.hide(cover, secret)
            _, rec = model.recover(stego, g=r)
        # save first images from the batch
        for i in range(min(cover.shape[0], n_samples - saved)):
            idx = saved + i + 1
            cv = to_0_1(cover[i:i+1]).cpu()
            st = to_0_1(stego[i:i+1]).cpu()
            rc = to_0_1(rec[i:i+1]).cpu()
            vutils.save_image(cv, STEGO_DIR / f"ep{epoch:03d}_sample{idx:02d}_cover.png")
            vutils.save_image(st, STEGO_DIR / f"ep{epoch:03d}_sample{idx:02d}_stego.png")
            vutils.save_image(rc, RECOVERED_DIR / f"ep{epoch:03d}_sample{idx:02d}_recovered.png")
            saved += 1
            if saved >= n_samples:
                model.train()
                return
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover_dir", type=str, default=None)
    parser.add_argument("--secret_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lambda-h", type=float, default=None)
    parser.add_argument("--lambda-r", type=float, default=None)
    parser.add_argument("--lambda-p", type=float, default=None)
    parser.add_argument("--save-samples-every", type=int, default=1, help="save sample images every N epochs (0 disables)")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda else DEVICE
    print(f"Training device: {device}")
    set_seed_local(SEED)

    # override config weights if provided
    if args.lambda_h is not None:
        import importlib
        import src.config as cfg
        cfg.LAMBDA_H = args.lambda_h
        print("Overrode LAMBDA_H to", cfg.LAMBDA_H)
    if args.lambda_r is not None:
        import src.config as cfg
        cfg.LAMBDA_R = args.lambda_r
        print("Overrode LAMBDA_R to", cfg.LAMBDA_R)
    if args.lambda_p is not None:
        import src.config as cfg
        cfg.LAMBDA_P = args.lambda_p
        print("Overrode LAMBDA_P to", cfg.LAMBDA_P)

    cover_dir = args.cover_dir or str(Path(__file__).resolve().parents[1] / "data" / "cover")
    secret_dir = args.secret_dir or str(Path(__file__).resolve().parents[1] / "data" / "secret")
    print("Using cover_dir:", cover_dir)
    print("Using secret_dir:", secret_dir)

    if args.debug:
        print("Debug mode enabled.")
        epochs = 1
        batch_size = min(4, args.batch_size)
    else:
        epochs = args.epochs
        batch_size = args.batch_size

    train_loader, val_loader = get_dataloaders(cover_dir=cover_dir, secret_dir=secret_dir, batch_size=batch_size, crop_size=TRAIN_CROP_SIZE, val_split=0.05, num_workers=args.num_workers)

    model = RISRANet(in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.amp else None
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    start_epoch = 1
    best_val_loss = float("inf")
    best_psnr_cs = 0.0
    no_improve_epochs = 0
    patience = args.patience

    if args.resume:
        print("Resuming from:", args.resume)
        start_epoch = load_checkpoint(model, optimizer, args.resume, map_location=device) + 1

    tb_dir = LOGS_DIR / "tensorboard"; tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    csv_path = LOGS_DIR / "train_log.csv"; write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["epoch","train_loss","val_loss","psnr_cover_stego","psnr_secret_rec","ssim_cover_stego","ssim_secret_rec","time_elapsed_s"])

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{epochs} -----------------------------")

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, args.amp, WAVELET, epoch, epochs, args.log_interval)

        val = validate(model, val_loader, device, WAVELET, use_amp=args.amp) if val_loader is not None else {"val_loss":0.0,"psnr_cover_stego":0.0,"psnr_secret_rec":0.0,"ssim_cover_stego":0.0,"ssim_secret_rec":0.0}
        val_loss = val["val_loss"]; psnr_cs = val["psnr_cover_stego"]; psnr_sr = val["psnr_secret_rec"]; ssim_cs = val["ssim_cover_stego"]; ssim_sr = val["ssim_secret_rec"]

        epoch_time = time.time() - epoch_start; elapsed_total = time.time() - t0
        print(f"Epoch {epoch} done — train_loss={train_loss:.4e}, val_loss={val_loss:.4e}, psnr_cs={psnr_cs:.2f}, psnr_sr={psnr_sr:.2f}, time={epoch_time:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PSNR/cover_stego", psnr_cs, epoch)
        writer.add_scalar("PSNR/secret_recovered", psnr_sr, epoch)
        writer.add_scalar("SSIM/cover_stego", ssim_cs, epoch)
        writer.add_scalar("SSIM/secret_recovered", ssim_sr, epoch)
        writer.flush()

        # save per-epoch checkpoint backup + latest/best
        per_epoch_checkpoint(model, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch, str(LATEST_CHECKPOINT))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, str(BEST_CHECKPOINT))
            print(f"New best model saved (val_loss={best_val_loss:.6e})")

        # save example images for inspection
        if args.save_samples_every > 0 and (epoch % args.save_samples_every == 0):
            save_sample_images(model, device, val_loader, epoch, n_samples=4)

        try:
            scheduler.step(val_loss)
        except Exception:
            pass

        if psnr_cs > best_psnr_cs + 1e-6:
            best_psnr_cs = psnr_cs; no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"No improvement in psnr_cs for {patience} epochs — early stopping.")
            break

        csv_writer.writerow([epoch, train_loss, val_loss, psnr_cs, psnr_sr, ssim_cs, ssim_sr, int(elapsed_total)])
        csv_file.flush()

    csv_file.close(); writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
