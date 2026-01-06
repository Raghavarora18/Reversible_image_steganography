
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from src.model.risranet import RISRANet
from src.losses import to_0_1, psnr, ssim
from src.config import DEVICE, STEGO_DIR, RECOVERED_DIR

def pil_to_tensor(img):
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return tf(img).unsqueeze(0)

def tensor_to_pil(t):
    t = t.detach().cpu().clamp(-1,1)
    t01 = (t+1)/2
    return T.ToPILImage()(t01.squeeze(0))

def load_checkpoint(model, path, device):
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
    elif isinstance(ck, dict):
        try:
            model.load_state_dict(ck)
        except Exception:
            model.load_state_dict(ck.get("model_state", ck))
    else:
        model.load_state_dict(ck)
    return ck.get("epoch", None) if isinstance(ck, dict) else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover", required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cover_p = Path(args.cover)
    secret_p = Path(args.secret)
    assert cover_p.exists() and secret_p.exists(), "Cover/secret must exist"

    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERED_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    model = RISRANet(in_channels=3).to(device)
    epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print("Loaded checkpoint:", args.checkpoint, "epoch:", epoch)

    cov = Image.open(cover_p).convert("RGB")
    sec = Image.open(secret_p).convert("RGB")
    cov_t = pil_to_tensor(cov).to(device)
    sec_t = pil_to_tensor(sec).to(device)

    with torch.no_grad():
        stego_t, g = model.hide(cov_t, sec_t)
        _, rec_t = model.recover(stego_t, g=g)

    # save
    outname = f"{cover_p.stem}__{secret_p.name}"
    stego_path = STEGO_DIR / outname
    rec_path = RECOVERED_DIR / outname
    tensor_to_pil(stego_t).save(stego_path)
    tensor_to_pil(rec_t).save(rec_path)

    # metrics
    psnr_cs = psnr(to_0_1(cov_t), to_0_1(stego_t))
    psnr_sr = psnr(to_0_1(sec_t), to_0_1(rec_t))
    ssim_cs = float(ssim(to_0_1(cov_t), to_0_1(stego_t)).detach().cpu().item())
    ssim_sr = float(ssim(to_0_1(sec_t), to_0_1(rec_t)).detach().cpu().item())

    print("Saved stego:", stego_path)
    print("Saved recovered:", rec_path)
    print(f"PSNR_cs: {psnr_cs:.2f}, PSNR_sr: {psnr_sr:.2f}, SSIM_cs: {ssim_cs:.4f}, SSIM_sr: {ssim_sr:.4f}")

if __name__ == "__main__":
    main()
