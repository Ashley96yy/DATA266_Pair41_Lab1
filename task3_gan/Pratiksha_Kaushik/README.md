# Task 3: Pratiksha Kaushik (CycleGAN, Monet ↔ Photo)

Unpaired image-to-image translation between Monet paintings and photos, using a CycleGAN trained from scratch. Pretrained networks are used only inside the metrics: Inception-v3 for FID/KID and AlexNet for LPIPS.

Personal parameters: `SID4=9002 | SEED=9002 | SLICE=2 | HP_ID=2 | CLS_A=2 | CLS_B=3`. All runs used an NVIDIA RTX 4090 (Python 3.10, torch 2.1.2).

## Final result

| Model | Photo→Monet FID | Monet→Photo FID | Submitted FID | Submitted MiFID | Score (FID+MiFID)/2 ↓ |
|---|---|---|---|---|---|
| v1 (baseline) | 101.51 | 102.99 | 102.25 | 0.4168 | 51.33 |
| **v3 (final)** | **100.54** | **99.58** | **100.06** | **0.4135** | **50.24** |

These scores come from the course evaluation script, using the first 300 files in each direction. See [RESULTS.md](RESULTS.md) for the full write-up.

## Two model versions

| | v1 (baseline) | v3 (final) |
|---|---|---|
| Notebook | [src/Improvements/task3-GAN.ipynb](src/Improvements/task3-GAN.ipynb) | [src/Pratiksha_cyclegan_monet_v3.ipynb](src/Pratiksha_cyclegan_monet_v3.ipynb) |
| Generator | ResNet, 9 blocks (11.38M) | same |
| Discriminator | 70×70 PatchGAN, 1 scale | PatchGAN at **2 scales** |
| Loss | LSGAN + 10·cycle + 5·identity | LSGAN + 10·cycle + **1**·identity |
| Training | 50 epochs / 40k steps, batch 4 | 80 epochs / 64k steps, batch 4 |
| Weights used | EMA, epoch 45 | EMA averaged over epochs 60/65/70 |
| Post-processing | none | colour calibration (Monet→photo only) |
| Official 300 photos in training | yes | no (excluded, so there's no leakage) |

Both versions use DiffAugment (colour + translation), an image pool of 50, EMA weights (decay 0.999) and bf16.

v3 settings came from a 15-epoch screening of 4 configurations. The 2-scale discriminator was the change that helped: `ms_id1` beat the base by 1.37 points.

## Folder layout

```
Pratiksha_Kaushik/
├── README.md                  this file
├── RESULTS.md                 main write-up: v1 vs v3, screening, per-epoch scores, metrics, stability, efficiency
├── METRICS .md                auto-generated metrics for v1 (note the space in the file name)
├── failure_analysis.md        7 failure categories for v3 (night scenes, skies, fine texture, steganography…)
├── metric_report.csv          v1 vs v3 metrics side by side
├── submission.csv             course-format CSV (ID,FID,MiFID); see the note below
├── submission_official_v3.csv v3 official submission (FID 100.06, MiFID 0.4135)
├── RUN_LOG (1).txt            full v3 log: screening, final run, evaluation, export
├── configs/
│   ├── config.json            base training config (notebook overrides λ_id / n_scales per run)
│   └── metrics.json           v1 metrics + stability stats (source of METRICS .md)
├── src/
│   ├── Pratiksha_cyclegan_monet_v3.ipynb   final notebook (sections 0–19)
│   └── Improvements/task3-GAN.ipynb        v1 baseline notebook ("task3-GAN 2.ipynb" is an identical copy)
├── checkpoints/               (git-ignored)
│   ├── best_ema.pt            v3 final generator weights (91 MB)
│   └── last.pt                full training state (514 MB)
├── data_processed/            (git-ignored)
│   └── train_history.csv      per-step losses and gradient norms
└── outputs/
    ├── eval_history.csv       FID per checkpoint (every 5 epochs)
    ├── figures/               data samples, loss and stability curves, qualitative results, worst cases
    ├── human_audit/           30-sample blinded audit: audit_sheet.html, images/, rater1/2.csv, _key.csv
    ├── pred_A2B/              empty (.gitkeep): 300 Photo→Monet predictions not copied here
    └── pred_B2A/              empty (.gitkeep): 7,038 Monet→Photo predictions not copied here
```

## Notebook walkthrough (v3)

1. **Setup:** personal params, dataset discovery (`monet_jpg/`, `photo_jpg/`), run log, config.
2. **Data:** 300 Monets, 7,038 photos, split into 5,772 train / 1,000 held-out. The 300 official photos are excluded.
3. **Model:** ResNet generator, multi-scale PatchGAN, DiffAugment, image pool, EMA, losses.
4. **Metrics:** a copy of the course FID/MiFID script, plus a separate validation set for checkpoint selection.
5. **Training:** 5a screening (4 configs × 15 epochs), then 5b the final 80-epoch run.
6. **Training behaviour and checkpoint selection:** the score plateaus after epoch 60, so epochs 60/65/70 are averaged.
7. **Colour calibration:** global colour correction for Monet→photo.
8. **Evaluation:** FID, KID, precision/recall, density/coverage, content cosine, LPIPS, cycle L1, memorisation.
9. **Qualitative results, cycle-consistency checks** (all PASS), **error analysis** and **efficiency**.
10. **Export:** predictions, official score, blinded human audit packet, METRICS.md, then a copy into the repo.

## Reproduce

1. Put the Kaggle "I'm Something of a Painter Myself" data (`monet_jpg/`, `photo_jpg/`) under the notebook's data root. It was `/app/content` on the training machine.
2. Open `src/Pratiksha_cyclegan_monet_v3.ipynb` and run it top to bottom with `QUICK=False`. A full run, including screening, takes about 6 h on an RTX 4090. The final run alone takes about 3.2 h.
3. With `resume=true`, training resumes from `checkpoints/last.pt` if it exists.

## Notes and open items

- **Human audit is not done.** `rater1.csv` and `rater2.csv` are still blank. To finish it, fill them in using `audit_sheet.html`, then rerun notebook §16.
- **Kaggle leaderboard:** no public or private score has been recorded yet.
- **Mismatched `submission.csv`:** it reports FID 82.57 / MiFID 0.4074, which doesn't match the v3 numbers in RESULTS.md or the run log (100.06 / 0.4135). Check which run produced it before submitting. `submission_official_v3.csv` matches v3.
- **`METRICS .md` and `configs/metrics.json` describe v1**, not v3. RESULTS.md has the v3 numbers.
- **Duplicate files:** many files with a ` 2` / ` 3` suffix (in figures, human_audit and Improvements) are byte-identical duplicates (Finder copies).
- **RESULTS.md file paths** (`data266_cyclegan_v3/...`, `FAILURES.md`, `screening.csv`) point to the training machine, not this folder.
