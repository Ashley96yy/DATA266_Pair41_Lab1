"""Local copy of the course evaluation script (task3_gan/Part3_Evaluation_Script.ipynb) as a .py file.

Folder layout expected under --root:
    monet_jpg/   real Monet            (domain A)
    photo_jpg/   real photos           (domain B)
    pred_A2B/    generated photos      (Monet -> Photo)
    pred_B2A/    generated Monets      (Photo -> Monet)

Takes the first N files by name in each folder (default 300), computes FID and the
paired cosine distance ("MiFID" in the course script) for both directions, averages
them and writes submission.csv.

Usage:
    python evaluate_local.py --root /app/content/data266_cyclegan_v3 \
        --monet /app/content/monet_jpg --photo /app/content/photo_jpg
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def list_images(folder):
    paths = []
    for ext in (".jpg", ".jpeg", ".png"):
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def take_n(paths, n):
    return paths if n is None else paths[:min(n, len(paths))]


def get_inception_model():
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    net.fc = nn.Identity()
    return net.to(device).eval()


@torch.no_grad()
def get_activations(model, paths, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(model, real_paths, gen_paths, batch_size=32):
    n = min(len(real_paths), len(gen_paths))
    real_paths, gen_paths = sorted(real_paths)[:n], sorted(gen_paths)[:n]
    real_act = get_activations(model, real_paths, batch_size)
    gen_act = get_activations(model, gen_paths, batch_size)
    fid = frechet_distance(real_act.mean(0), np.cov(real_act, rowvar=False),
                           gen_act.mean(0), np.cov(gen_act, rowvar=False))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(n)]))
    return fid, mifid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains pred_A2B/ and pred_B2A/")
    ap.add_argument("--monet", help="real Monet folder (default: ROOT/monet_jpg)")
    ap.add_argument("--photo", help="real photo folder (default: ROOT/photo_jpg)")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    real_monet_dir = args.monet or os.path.join(args.root, "monet_jpg")
    real_photo_dir = args.photo or os.path.join(args.root, "photo_jpg")
    gen_a2b_dir = os.path.join(args.root, "pred_A2B")
    gen_b2a_dir = os.path.join(args.root, "pred_B2A")
    for d in (real_monet_dir, real_photo_dir, gen_a2b_dir, gen_b2a_dir):
        assert os.path.isdir(d), f"Missing folder: {d}"

    real_monet = take_n(list_images(real_monet_dir), args.n_eval)
    real_photo = take_n(list_images(real_photo_dir), args.n_eval)
    gen_a2b = take_n(list_images(gen_a2b_dir), args.n_eval)
    gen_b2a = take_n(list_images(gen_b2a_dir), args.n_eval)
    print(f"Real Monet {len(real_monet)} | Gen Monet (B2A) {len(gen_b2a)}")
    print(f"Real Photo {len(real_photo)} | Gen Photo (A2B) {len(gen_a2b)}")

    model = get_inception_model()
    fid_b2a, mifid_b2a = calculate_fid_mifid(model, real_monet, gen_b2a)
    print(f"[Photo->Monet] FID={fid_b2a:.3f}  MiFID={mifid_b2a:.4f}")
    fid_a2b, mifid_a2b = calculate_fid_mifid(model, real_photo, gen_a2b)
    print(f"[Monet->Photo] FID={fid_a2b:.3f}  MiFID={mifid_a2b:.4f}")

    sub = pd.DataFrame([{"ID": 1, "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2}])
    sub.to_csv(args.out, index=False)
    print(sub.to_string(index=False))
    print(f"score (FID + MiFID) / 2 = {(sub.FID[0] + sub.MiFID[0]) / 2:.4f}")


if __name__ == "__main__":
    main()
