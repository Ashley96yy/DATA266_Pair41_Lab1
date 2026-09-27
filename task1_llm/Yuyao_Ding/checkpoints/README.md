# Task 1 - Yuyao Ding - checkpoints

[Download checkpoints from Google Drive](https://drive.google.com/drive/folders/1Aq5tUVV39Tkx4q85AOq1TgamUxZ0IHev?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task1_llm/Yuyao_Ding/checkpoints/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`checkpoints/checkpoints/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

The current RTX 4090 report and default demo use:

```text
train_20260927T000510Z_b9df2210/epoch_010.pt
```

Epoch 10 is both the best and final checkpoint. Restore the matching
`vocabulary.json` from the [processed-data download](../data_processed/README.md)
before running the demo. Historical Mac/smoke checkpoints, when downloaded, must
remain in their original run directories and must not replace this reported run.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
