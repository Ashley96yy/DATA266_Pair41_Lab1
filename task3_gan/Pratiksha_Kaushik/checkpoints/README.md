# Checkpoints - Task 3 (Pratiksha Kaushik)

The model files are too large for the repo, so they are stored on Google Drive:

**Download:** https://drive.google.com/drive/folders/1CGzq0IoI6pX0s4msiYyyfpYXTPhZJhHp

Put the downloaded files in this folder.

| File | What it is | Size |
|---|---|---|
| `last_v3_epoch80.pt` | Full v3 training state at epoch 80: both generators, both 2-scale discriminators, EMA generators, optimizers, schedulers, per-step history, eval history and config | 514 MB |
| `best_ema_v1_epoch45.pt` | v1 baseline EMA generators at epoch 45 | 91 MB |

## Final v3 model

The model actually used for `submission.csv`, the Kaggle submission and every v3 metric is in a separate Drive folder:

**Download:** https://drive.google.com/drive/folders/1mOjhaPIFw9ML5e-Ts3aGs_CDKdgv31an?usp=drive_link

| File | What it is |
|---|---|
| `best_ema.pt` | EMA generators averaged over epochs 60, 65 and 70 of the final run |
| `color_cal.pt` | global colour calibration applied to Monet→photo outputs |

`last_v3_epoch80.pt` above is the epoch-80 training state, not this averaged model.

Load the generators with:

```python
from src.cyclegan_models import load_generators
G_P2M, G_M2P = load_generators("checkpoints/last_v3_epoch80.pt")
```


