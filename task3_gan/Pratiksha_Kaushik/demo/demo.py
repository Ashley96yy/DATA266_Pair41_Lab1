"""Demo: translate one image with the final v3 CycleGAN and save input | translation | cycle reconstruction.

Directions (same as the course script, A = Monet, B = photo):
    B2A: photo -> Monet   (default)
    A2B: Monet -> photo   (uses the colour calibration, like the submitted model)

Usage (from task3_gan/Pratiksha_Kaushik/):
    python demo/demo.py --image my_photo.jpg
    python demo/demo.py --image some_monet.jpg --direction A2B --output demo_photo.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))   # model code lives in src/
from cyclegan_models import load_generators  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
CKPT_DIR = HERE.parent / "checkpoints"


def to_tensor(img, size=256):
    # resize the shorter side to 256 and centre-crop, so any photo works
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(127.5).sub(1)
    return img, x.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def to_image(G, x, cal=None):
    y = (G(x).float().clamp(-1, 1) + 1) * 127.5
    if cal is not None:
        mu_g, sd_g, mu_r, sd_r = (t.to(y.device).view(1, 3, 1, 1) for t in cal)
        y = (y - mu_g) / sd_g * sd_r + mu_r
    y = y.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--direction", choices=["B2A", "A2B"], default="B2A",
                    help="B2A = photo -> Monet (default), A2B = Monet -> photo")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best_ema_v3_final.pt"))
    ap.add_argument("--color-cal", default=str(CKPT_DIR / "color_cal.pt"))
    ap.add_argument("--output", default="demo_output.png")
    a = ap.parse_args()

    G_P2M, G_M2P = load_generators(a.ckpt, DEVICE)
    cal = None
    if a.direction == "A2B" and Path(a.color_cal).exists():
        cal = torch.load(a.color_cal, map_location="cpu", weights_only=False)["M2P"]

    inp, x = to_tensor(Image.open(a.image).convert("RGB"))
    if a.direction == "B2A":
        fwd, back, labels = G_P2M, G_M2P, ("photo (input)", "-> Monet", "-> photo (cycle)")
        out = to_image(fwd, x)
    else:
        fwd, back, labels = G_M2P, G_P2M, ("Monet (input)", "-> photo", "-> Monet (cycle)")
        out = to_image(fwd, x, cal)
    _, y = to_tensor(out)
    rec = to_image(back, y)

    panel = Image.new("RGB", (256 * 3 + 16, 256 + 24), "white")
    draw = ImageDraw.Draw(panel)
    for i, (im, lab) in enumerate(zip((inp, out, rec), labels)):
        panel.paste(im, (i * (256 + 8), 24))
        draw.text((i * (256 + 8) + 4, 5), lab, fill="black")
    panel.save(a.output)
    out.save(Path(a.output).with_name(Path(a.output).stem + "_translation.jpg"), quality=95)
    print(f"device {DEVICE} | {a.direction} | saved {a.output} and the translation alone next to it")


if __name__ == "__main__":
    main()
