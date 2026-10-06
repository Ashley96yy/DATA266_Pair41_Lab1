# pred_A2B - Monet → Photo predictions (Pratiksha Kaushik)

The images are too large for the repo, so they are stored on Google Drive:

**Download:** https://drive.google.com/drive/folders/1GExvGuG6bCmy1YvEYH2r8PaEPvyDrr0E

- 300 generated photos, one per Monet painting in `monet_jpg/`. Each file has the same name as its input painting, saved as JPEG (quality 95, 256×256).
- Made by the final v3 model (`src/Pratiksha_cyclegan_monet_v3.ipynb`, section 14): EMA generator averaged over epochs 60/65/70, with colour calibration.
- These are the files the course evaluation script scores as "Monet → Photo" (first 300 by name).

To score locally, put the downloaded images in this folder and run:

```bash
python evaluate_local.py --root outputs --monet /path/to/monet_jpg --photo /path/to/photo_jpg
```
