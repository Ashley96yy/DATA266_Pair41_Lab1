# Task 3 - Yuyao Ding - checkpoints

[Download checkpoints from Google Drive](https://drive.google.com/drive/folders/1fkqIV4PLRg7x7v6cEMiWDrrx5O2ATTAV?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task3_gan/Yuyao_Ding/checkpoints/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`checkpoints/checkpoints/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

The current nine-block CycleGAN uses:

```text
task3_formal_20260927T075834Z_d335ae80/
  best.pt
  best.json
  last.pt
  last.json
```

`best.pt` is the selected epoch-70 checkpoint used by the demo and reports;
`last.pt` is the final epoch-100 training state. **Keep each `.pt` file with its
matching `.json` sidecar**: the loader verifies the checkpoint checksum.

The older six-block comparison model is kept separately under
`task3_formal_20260927T043055Z_42bb4f28/` (`best.pt` and `best.json`, epoch 39).
Do not mix these two runs. A single-image demo also needs an input image;
full evaluation needs the [saved split](../data_processed/README.md), raw images
and metric-network weights described in the task guide.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
