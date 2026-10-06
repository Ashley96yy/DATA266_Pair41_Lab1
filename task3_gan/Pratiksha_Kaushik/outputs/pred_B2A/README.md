# pred_B2A - Photo → Monet predictions (Pratiksha Kaushik)

The images are too large for the repo, so they are stored on Google Drive:

**Download:** https://drive.google.com/drive/folders/1H3RnjVjtGxT_iCU0XmzZvRwEg55_ekQr

- 7,038 generated Monet-style images, one per photo in `photo_jpg/`. Each file has the same name as its input photo, saved as JPEG (quality 95, 256×256).
- Made by the final v3 model (`src/Pratiksha_cyclegan_monet_v3.ipynb`, section 14): EMA generator averaged over epochs 60/65/70.
- The course evaluation script scores the first 300 by name as "Photo → Monet". Those 300 photos were left out of training.

To score locally, put the downloaded images in this folder (and the `pred_A2B` images in `../pred_A2B/`) and run:

```bash
python evaluate_local.py --root outputs --monet /path/to/monet_jpg --photo /path/to/photo_jpg
```
