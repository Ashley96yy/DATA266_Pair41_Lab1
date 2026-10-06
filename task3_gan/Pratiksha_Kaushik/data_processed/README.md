# Processed data - Task 3 (Pratiksha Kaushik)

Per-step training records written by the notebooks. The image data itself (300 Monet and 7,038 photo JPEGs) is the Kaggle competition data and is not stored here; the notebook loads it from `monet_jpg/` and `photo_jpg/` and makes the split in memory (seed 9002: 5,772 training photos, 1,000 held-out photos, the 300 course-scored photos excluded from training).

| File | What it is | Rows | SHA-256 |
|---|---|---|---|
| `train_history_v3.csv` | final v3 run: loss terms (G, D, adversarial, cycle, identity), gradient norms and mean discriminator outputs for every step | 64,000 | `f0bb05fd101cd3815ce224d800e699bd7724e4787e5c6e2ff1c311681a590438` |
| `train_history_v1.csv` | the same for the v1 baseline | 40,000 | `9edca879815e7c17eb9d3d751ce7688d358b73ed8a9a6f11f6e906b801f6c69e` |

These are the data behind `outputs/figures/loss_curves.png`, `outputs/figures/stability_curves.png` and the loss, gradient-norm and NaN rows of `full_metrics_report.csv`. The v3 file was exported from the `HIST` record inside `checkpoints/last_v3_epoch80.pt`.
