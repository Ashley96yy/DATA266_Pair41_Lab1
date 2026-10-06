# METRICS — CycleGAN Monet ↔ Photo

SID4=9002 | SEED=9002 | SLICE=2 | HP_ID=2 | CLS_A=2 | CLS_B=3
Best EMA checkpoint: epoch 45 of 50 | GPU: NVIDIA GeForce RTX 4090

## Distribution metrics (Inception-v3 pool3, 2048-d)

| Metric | Photo → Monet | Monet → Photo |
|---|---|---|
| FID ↓ | 94.67 | 86.15 |
| KID ×10³ ↓ (mean ± std) | 19.48 ± 2.06 | 18.55 ± 2.08 |
| Precision ↑ | 0.4320 | 0.6833 |
| Recall ↑ | 0.6567 | 0.4500 |
| Density ↑ | 0.3158 | 0.7300 |
| Coverage ↑ | 0.7900 | 0.4540 |
| Content cosine (input vs translation) ↑ | 0.7655 | 0.8164 |
| LPIPS input vs translation | 0.3718 | 0.3016 |
| Cycle reconstruction L1 ↓ | 0.0934 (photo cycle) | 0.0678 (Monet cycle) |
| LPIPS input vs reconstruction ↓ | 0.2077 | 0.2735 |

Reference points: FID photo-vs-photo split floor 57.32; unrelated-pair L1 0.5742, LPIPS 0.8114, content cosine 0.5689. MiFID (Kaggle-style, vs training Monets) 94.67, memorisation distance 0.2612.

## Training losses (mean over final 25% of steps)

| Term | Value |
|---|---|
| Discriminator total | 0.3262 |
| Generator adversarial (both) | 0.9729 |
| Cycle-consistency L1 (both, unweighted) | 0.1255 (first 25%: 0.2990) |
| Identity L1 (both, unweighted) | 0.1114 |
| λ_cyc / λ_id | 10.0 / 5.0 |

Curves: `figures/training_curves.png`

## Stability

| Metric | G | D |
|---|---|---|
| Grad-norm median | 21.725 | 11.799 |
| Grad-norm p99 | 53.156 | 31.390 |
| Grad-norm max | 198.295 | 245.876 |
| NaN/Inf steps | 0 (of 40000) | |

## Human audit (30 samples, 2 raters, 1–5)

Not run for v1. The blinded human audit was done on the final v3 model; see `results.md` section 9 (overall 3.49 / 5).

## Efficiency

| Metric | Value |
|---|---|
| Params per generator / discriminator | 11.38M / 2.76M |
| Params total | 28.29M |
| Training time | 1.83 h (40000 steps, batch 4) |
| Training throughput | 24.2 img/s per domain |
| Inference throughput (bs 16, median of 5) | 523.8 ± 4.7 img/s |
| Peak GPU memory train / inference | 18.51 GB / 2.44 GB |
