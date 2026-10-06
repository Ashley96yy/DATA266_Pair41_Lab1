"""Full Task 3 metrics for a trained CycleGAN checkpoint.

Same formulas as section 9 of Pratiksha_cyclegan_monet_v3.ipynb:
FID, KID, precision/recall/density/coverage, MiFID + memorisation distance,
content cosine, LPIPS (input vs translation and input vs reconstruction), cycle L1.

Inception features: torch-fidelity "inception-v3-compat" (same weights as TF FID).
LPIPS: AlexNet. These pretrained nets are only used for measuring, not for training.

Usage:
    python src/full_metrics.py --ckpt checkpoints/last_v3_epoch80.pt \
        --monet /path/monet_jpg --photo /path/photo_jpg --out outputs/full_metrics_rerun.json

Held-out photos: same rule as the notebook (seed 9002, 1,000 photos not used for training).
Pass --heldout-list with a text file of filenames if you want the exact notebook split.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyclegan_models import load_generators  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 9002


def load_folder(folder, files=None, size=256):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if files is not None:
        keep = set(files)
        paths = [p for p in paths if p.name in keep]
    arr = [np.asarray(Image.open(p).convert("RGB").resize((size, size))) for p in paths]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous(), paths


def to_float(u8):
    return u8.float() / 127.5 - 1


@torch.no_grad()
def translate(G, imgs_u8, bs=32):
    out = []
    for i in range(0, len(imgs_u8), bs):
        y = G(to_float(imgs_u8[i:i + bs].to(DEVICE)))
        out.append(((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
    return torch.cat(out)


def get_inception():
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(DEVICE).eval()


@torch.no_grad()
def inception_feats(net, imgs_u8, bs=100):
    return torch.cat([net(imgs_u8[i:i + bs].to(DEVICE))[0].double().cpu()
                      for i in range(0, len(imgs_u8), bs)]).numpy()


def fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def kid(f1, f2, n_subsets=100, subset_size=None, seed=SEED):
    rs = np.random.RandomState(seed)
    m = subset_size or min(len(f1), len(f2), 1000)
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rs.choice(len(f1), m, replace=False)]
        y = f2[rs.choice(len(f2), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def prdc(real, fake, k=5):
    real = torch.from_numpy(real).float().to(DEVICE)
    fake = torch.from_numpy(fake).float().to(DEVICE)
    d_rr, d_ff, d_rf = torch.cdist(real, real), torch.cdist(fake, fake), torch.cdist(real, fake)
    r_rad = d_rr.kthvalue(k + 1, dim=1).values
    f_rad = d_ff.kthvalue(k + 1, dim=1).values
    return dict(
        precision=(d_rf < r_rad[:, None]).any(0).float().mean().item(),
        recall=(d_rf < f_rad[None, :]).any(1).float().mean().item(),
        density=((d_rf < r_rad[:, None]).float().sum(0) / k).mean().item(),
        coverage=(d_rf.min(1).values < r_rad).float().mean().item(),
    )


def mifid(gen, train, fid_value, eps=0.1):
    g = gen / np.linalg.norm(gen, axis=1, keepdims=True)
    t = train / np.linalg.norm(train, axis=1, keepdims=True)
    d = float((1 - g @ t.T).min(1).mean())
    return fid_value / ((d if d < eps else 1.0) + 1e-15), d


def cos_rows(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def l1_px(a, b):
    return (a.float() - b.float()).abs().mean(dim=(1, 2, 3)) / 127.5


@torch.no_grad()
def lpips_pairs(net, a_u8, b_u8, bs=50):
    return torch.cat([net(to_float(a_u8[i:i + bs].to(DEVICE)), to_float(b_u8[i:i + bs].to(DEVICE))).flatten().cpu()
                      for i in range(0, len(a_u8), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--monet", required=True, help="folder with real Monet jpgs")
    ap.add_argument("--photo", required=True, help="folder with real photo jpgs")
    ap.add_argument("--heldout-list", help="txt file with held-out photo filenames, one per line")
    ap.add_argument("--n-photos", type=int, default=1000)
    ap.add_argument("--out", default="outputs/full_metrics_rerun.json")
    args = ap.parse_args()

    G_P2M, G_M2P = load_generators(args.ckpt, DEVICE)
    monet, _ = load_folder(args.monet)
    if args.heldout_list:
        names = [l.strip() for l in open(args.heldout_list) if l.strip()]
        photo, _ = load_folder(args.photo, names)
    else:
        photo_all, _ = load_folder(args.photo)
        idx = np.random.RandomState(SEED).choice(len(photo_all), args.n_photos, replace=False)
        photo = photo_all[np.sort(idx)]

    fake_monet, fake_photo = translate(G_P2M, photo), translate(G_M2P, monet)
    rec_photo, rec_monet = translate(G_M2P, fake_monet), translate(G_P2M, fake_photo)

    inc = get_inception()
    F_in_P, F_in_M = inception_feats(inc, photo), inception_feats(inc, monet)
    F_fake_M, F_fake_P = inception_feats(inc, fake_monet), inception_feats(inc, fake_photo)

    M = {}
    M["FID_photo2monet"] = fid(F_fake_M, F_in_M)
    M["FID_monet2photo"] = fid(F_fake_P, F_in_P)
    ss = min(100, len(F_in_M), len(F_in_P))
    M["KID_photo2monet"], M["KID_photo2monet_std"] = kid(F_fake_M, F_in_M, subset_size=ss)
    M["KID_monet2photo"], M["KID_monet2photo_std"] = kid(F_fake_P, F_in_P, subset_size=ss)
    h = len(F_in_P) // 2
    M["FID_floor_photo_vs_photo"] = fid(F_in_P[:h], F_in_P[h:])
    for k, v in prdc(F_in_M, F_fake_M).items():
        M[f"{k}_photo2monet"] = v
    for k, v in prdc(F_in_P, F_fake_P).items():
        M[f"{k}_monet2photo"] = v
    M["MiFID_photo2monet"], M["memorization_dist"] = mifid(F_fake_M, F_in_M, M["FID_photo2monet"])

    M["content_cos_photo2monet"] = float(cos_rows(F_in_P, F_fake_M).mean())
    M["content_cos_monet2photo"] = float(cos_rows(F_in_M, F_fake_P).mean())
    M["content_cos_unrelated_baseline"] = float(cos_rows(F_in_P, np.roll(F_in_P, 1, axis=0)).mean())

    M["cycle_L1_photo"] = float(l1_px(rec_photo, photo).mean())
    M["cycle_L1_monet"] = float(l1_px(rec_monet, monet).mean())
    M["cycle_L1_unrelated_baseline"] = float(l1_px(photo, photo.roll(1, 0)).mean())

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
    M["LPIPS_input_vs_translation_photo2monet"] = float(lpips_pairs(lp, photo, fake_monet).mean())
    M["LPIPS_input_vs_translation_monet2photo"] = float(lpips_pairs(lp, monet, fake_photo).mean())
    M["LPIPS_input_vs_reconstruction_photo"] = float(lpips_pairs(lp, photo, rec_photo).mean())
    M["LPIPS_input_vs_reconstruction_monet"] = float(lpips_pairs(lp, monet, rec_monet).mean())
    M["LPIPS_unrelated_baseline"] = float(lpips_pairs(lp, photo[:500], photo.roll(1, 0)[:500]).mean())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(M, open(args.out, "w"), indent=2)
    for k, v in M.items():
        print(f"{k:42s} {v:.4f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
