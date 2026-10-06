"""All Task 3 metrics on the course protocol (first 300 files by sorted name, both directions).

Directions follow the course script: A = Monet, B = photo.
    A to B: Monet -> photo  (inputs: first 300 Monets,  outputs: pred_A2B, reference: first 300 real photos)
    B to A: photo -> Monet  (inputs: first 300 photos,  outputs: pred_B2A, reference: first 300 real Monets)

Metrics per direction:
    evaluated image count, FID and course MiFID (same code as the course script),
    KID (raw scale), precision / recall and density / coverage (k = 5),
    content-preservation cosine (input vs its own translation),
    LPIPS (input vs translation, input vs cycle reconstruction),
    cycle-reconstruction L1 (needs the generators: --ckpt),
    translation images per second (needs the generators: --ckpt).

Every feature-based metric uses the course torchvision Inception-v3 (fc = Identity, 2048-d),
so all rows in this table share one feature extractor.

Usage:
    python src/course_protocol_metrics.py \
        --monet /path/monet_jpg --photo /path/photo_jpg \
        --pred-a2b outputs/pred_A2B --pred-b2a outputs/pred_B2A \
        --ckpt checkpoints/best_ema.pt --out outputs/course_protocol_metrics.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyclegan_models import load_generators  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 9002
N_EVAL = 300

INCEPTION_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def first_n(folder, n=N_EVAL):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return paths[:n]


def match_outputs(inputs, out_dir):
    """Output files are saved under the input's file stem (notebook section 14)."""
    out = {p.stem: p for p in Path(out_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    missing = [p.name for p in inputs if p.stem not in out]
    if missing:
        raise SystemExit(f"{len(missing)} inputs have no translation in {out_dir}, e.g. {missing[:3]}")
    return [out[p.stem] for p in inputs]


def get_inception():
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    net.fc = nn.Identity()
    return net.to(DEVICE).eval()


@torch.no_grad()
def feats(net, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEVICE)
        out.append(net(x).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    covmean = scipy.linalg.sqrtm(s1.dot(s2))
    if isinstance(covmean, tuple):   # older SciPy returns (sqrtm, error)
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))


def course_fid_mifid(real, gen):
    n = min(len(real), len(gen))
    real, gen = real[:n], gen[:n]
    fid = frechet_distance(real.mean(0), np.cov(real, rowvar=False), gen.mean(0), np.cov(gen, rowvar=False))
    mifid = float(np.mean([cosine(real[i], gen[i]) for i in range(n)]))
    return fid, mifid


def kid(f1, f2, n_subsets=100, subset_size=100, seed=SEED):
    rs = np.random.RandomState(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rs.choice(len(f1), m, replace=False)]
        y = f2[rs.choice(len(f2), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals)), m


def prdc(real, fake, k=5):
    real, fake = torch.from_numpy(real).float(), torch.from_numpy(fake).float()
    d_rr, d_ff, d_rf = torch.cdist(real, real), torch.cdist(fake, fake), torch.cdist(real, fake)
    r_rad = d_rr.kthvalue(k + 1, dim=1).values
    f_rad = d_ff.kthvalue(k + 1, dim=1).values
    return dict(precision=(d_rf < r_rad[:, None]).any(0).float().mean().item(),
                recall=(d_rf < f_rad[None, :]).any(1).float().mean().item(),
                density=((d_rf < r_rad[:, None]).float().sum(0) / k).mean().item(),
                coverage=(d_rf.min(1).values < r_rad).float().mean().item())


def load_u8(paths, size=256):
    arr = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)) for p in paths]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def to_float(u8):
    return u8.float() / 127.5 - 1


@torch.no_grad()
def translate(G, u8, bs=16):
    out = []
    for i in range(0, len(u8), bs):
        y = G(to_float(u8[i:i + bs].to(DEVICE)))
        out.append(((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
    return torch.cat(out)


@torch.no_grad()
def images_per_second(G, u8, bs=16, repeats=3):
    x = to_float(u8[:bs].to(DEVICE))
    G(x)  # warm-up
    rates = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        translate(G, u8, bs)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        rates.append(len(u8) / (time.perf_counter() - t))
    return float(np.median(rates))


def l1_px(a, b):
    return (a.float() - b.float()).abs().mean(dim=(1, 2, 3)) / 127.5


@torch.no_grad()
def lpips_pairs(net, a, b, bs=25):
    return torch.cat([net(to_float(a[i:i + bs].to(DEVICE)), to_float(b[i:i + bs].to(DEVICE))).flatten().cpu()
                      for i in range(0, len(a), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monet", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--pred-a2b", required=True, help="Monet -> photo outputs")
    ap.add_argument("--pred-b2a", required=True, help="photo -> Monet outputs")
    ap.add_argument("--ckpt", help="generator checkpoint for cycle L1 and images/s (optional)")
    ap.add_argument("--out", default="outputs/course_protocol_metrics.csv")
    a = ap.parse_args()

    monet_in, photo_in = first_n(a.monet), first_n(a.photo)
    a2b_out = match_outputs(monet_in, a.pred_a2b)   # translations of the 300 Monet inputs
    b2a_out = match_outputs(photo_in, a.pred_b2a)   # translations of the 300 photo inputs
    print(f"device {DEVICE} | Monet {len(monet_in)} | photo {len(photo_in)}")

    inc = get_inception()
    F_monet, F_photo = feats(inc, monet_in), feats(inc, photo_in)
    F_a2b, F_b2a = feats(inc, a2b_out), feats(inc, b2a_out)

    rows = {}
    for name, F_in, F_out, F_ref in (("A to B (Monet -> photo)", F_monet, F_a2b, F_photo),
                                     ("B to A (photo -> Monet)", F_photo, F_b2a, F_monet)):
        fid, mifid = course_fid_mifid(F_ref, F_out)
        k_mean, k_std, k_m = kid(F_out, F_ref)
        pr = prdc(F_ref, F_out)
        cos = (F_in / np.linalg.norm(F_in, axis=1, keepdims=True) * F_out / np.linalg.norm(F_out, axis=1, keepdims=True)).sum(1)
        rows[name] = {
            "Evaluated image count": len(F_out),
            "FID": round(fid, 4),
            "Course MiFID": round(mifid, 6),
            "KID and scale": f"{k_mean:.6f} raw (± {k_std:.6f}, 100 subsets of {k_m})",
            "Generative precision": round(pr["precision"], 4),
            "Generative recall": round(pr["recall"], 4),
            "Density and coverage if used": f"{pr['density']:.4f} / {pr['coverage']:.4f}",
            "Content-preservation cosine similarity": round(float(cos.mean()), 6),
        }

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
    monet_u8, photo_u8 = load_u8(monet_in), load_u8(photo_in)
    a2b_u8, b2a_u8 = load_u8(a2b_out), load_u8(b2a_out)
    lp_a2b = float(lpips_pairs(lp, monet_u8, a2b_u8).mean())
    lp_b2a = float(lpips_pairs(lp, photo_u8, b2a_u8).mean())

    if a.ckpt:
        G_P2M, G_M2P = load_generators(a.ckpt, DEVICE)
        rec_monet = translate(G_P2M, a2b_u8)    # Monet -> photo (saved) -> Monet
        rec_photo = translate(G_M2P, b2a_u8)    # photo -> Monet (saved) -> photo
        cyc_a = float(l1_px(rec_monet, monet_u8).mean())
        cyc_b = float(l1_px(rec_photo, photo_u8).mean())
        lp_rec_a = float(lpips_pairs(lp, monet_u8, rec_monet).mean())
        lp_rec_b = float(lpips_pairs(lp, photo_u8, rec_photo).mean())
        ips_a = images_per_second(G_M2P, monet_u8)
        ips_b = images_per_second(G_P2M, photo_u8)
        dev = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else DEVICE.type
        rows["A to B (Monet -> photo)"].update({
            "Cycle-reconstruction L1": round(cyc_a, 6),
            "LPIPS and image comparison": f"input vs translation {lp_a2b:.4f}; input vs reconstruction {lp_rec_a:.4f} (AlexNet)",
            "Translation images per second": f"{ips_a:.2f} ({dev}, batch 16)"})
        rows["B to A (photo -> Monet)"].update({
            "Cycle-reconstruction L1": round(cyc_b, 6),
            "LPIPS and image comparison": f"input vs translation {lp_b2a:.4f}; input vs reconstruction {lp_rec_b:.4f} (AlexNet)",
            "Translation images per second": f"{ips_b:.2f} ({dev}, batch 16)"})
    else:
        rows["A to B (Monet -> photo)"]["LPIPS and image comparison"] = f"input vs translation {lp_a2b:.4f} (AlexNet)"
        rows["B to A (photo -> Monet)"]["LPIPS and image comparison"] = f"input vs translation {lp_b2a:.4f} (AlexNet)"

    order = ["Evaluated image count", "FID", "Course MiFID", "KID and scale", "Generative precision",
             "Generative recall", "Density and coverage if used", "Cycle-reconstruction L1",
             "LPIPS and image comparison", "Content-preservation cosine similarity", "Translation images per second"]
    table = pd.DataFrame(rows).reindex(order).fillna("needs --ckpt")
    table.index.name = "Metric"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out)
    print(table.to_string())
    print("saved", a.out)


if __name__ == "__main__":
    main()
