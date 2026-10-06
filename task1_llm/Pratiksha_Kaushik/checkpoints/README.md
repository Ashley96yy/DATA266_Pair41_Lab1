# Checkpoints - Task 1 (Pratiksha Kaushik)

The checkpoint files are too large for the repo, so they are stored on Google Drive:

**Download:** https://drive.google.com/drive/folders/18IwhldDiOowviFXWqD22ZC2dRBwvGPEF?usp=drive_link

Put the downloaded files in this folder.

| File | What it is | Size | SHA-256 |
|---|---|---|---|
| `ckpt_best.pt` | model weights with the best validation loss (epoch 10, step 27,360, val CE 0.4742). Used for every number in `results.md`. | 102 MB | `4d9d958862777fe058443fc0f30d23cfc69ec451b4fc746a5650d8fedfbf864a` |
| `ckpt_last.pt` | full training state at the last epoch (model, optimizer, scheduler), for resuming training | 307 MB | `59f2908ffb333e4b34b4877a2e8935c2e95b4f68617682675852f93440777195` |

Both come from run `20261005-223946_L8H8C512T512` (8 layers, 8 heads, 512-d, context 512, 25.56M parameters). Because the best epoch was also the last one, both files hold the same model weights.

To check a download:

```bash
shasum -a 256 checkpoints/ckpt_best.pt checkpoints/ckpt_last.pt
```

To use the model without retraining, open `src/Task1_gpt.ipynb`, run the setup and model cells, and then run the evaluation and generation cells. They load `ckpt_best.pt`.
