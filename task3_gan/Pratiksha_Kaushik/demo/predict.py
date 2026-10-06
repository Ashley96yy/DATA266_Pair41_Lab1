"""Translate images with a saved CycleGAN checkpoint and write pred_A2B / pred_B2A folders.

A = Monet, B = photo (same as the course script):
    pred_A2B  <- Monet -> photo  (generator EMA_M2P)
    pred_B2A  <- photo -> Monet  (generator EMA_P2M)

Outputs are 256x256 JPEGs (quality 95) saved under the input's file name, like notebook section 14,
so evaluate_local.py can score them directly.

Usage:
    python demo/predict.py --ckpt checkpoints/best_ema_v3_final.pt --color-cal checkpoints/color_cal.pt \
        --monet /path/monet_jpg --photo /path/photo_jpg --out eval_final --n 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # model code lives in src/
from cyclegan_models import load_generators  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")


def list_images(folder, n=None):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return paths if n is None else paths[:n]


def apply_cal(y255, cal):
    # global per-channel colour correction, same as notebook section 8:
    # (x - mean_gen) / std_gen * std_real + mean_real
    mu_g, sd_g, mu_r, sd_r = (t.to(y255.device).view(1, 3, 1, 1) for t in cal)
    return (y255 - mu_g) / sd_g * sd_r + mu_r


@torch.no_grad()
def run(G, paths, out_dir, bs=16, size=256, cal=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(paths), bs):
        batch = paths[i:i + bs]
        x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)) for p in batch])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEVICE)
        y = (G(x).float().clamp(-1, 1) + 1) * 127.5
        if cal is not None:
            y = apply_cal(y, cal)
        y = y.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for p, img in zip(batch, y):
            Image.fromarray(img).save(out_dir / f"{p.stem}.jpg", format="JPEG", quality=95)
        print(f"\r{out_dir.name}: {min(i + bs, len(paths))}/{len(paths)}", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--monet", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", required=True, help="folder that will contain pred_A2B/ and pred_B2A/")
    ap.add_argument("--n", type=int, default=300, help="first N files per domain (course protocol = 300; 0 = all)")
    ap.add_argument("--color-cal", help="color_cal.pt; applied to Monet -> photo outputs like the final v3 model")
    a = ap.parse_args()
    n = a.n or None

    G_P2M, G_M2P = load_generators(a.ckpt, DEVICE)
    print("device", DEVICE, "| checkpoint", a.ckpt)
    out = Path(a.out)
    cal = None
    if a.color_cal:
        cal = torch.load(a.color_cal, map_location="cpu", weights_only=False)["M2P"]
        print("colour calibration on Monet -> photo:", a.color_cal)
    run(G_M2P, list_images(a.monet, n), out / "pred_A2B", cal=cal)
    run(G_P2M, list_images(a.photo, n), out / "pred_B2A")
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
