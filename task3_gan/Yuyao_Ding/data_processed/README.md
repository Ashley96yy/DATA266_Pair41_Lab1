# Task 3 - Yuyao Ding - data_processed

[Download data_processed from Google Drive](https://drive.google.com/drive/folders/1KWZo-O5WC11xQQ2eOS9T4UH_z6ZB2Upb?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task3_gan/Yuyao_Ding/data_processed/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`data_processed/data_processed/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

Restore this file directly into the destination directory:

```text
split_seed41.json
```

It records the train/validation/test split and keeps duplicate-photo groups in
the same partition. Preserve it unchanged for reproduction and full evaluation.
It does not contain the raw Monet/photo images; use the separate raw-data link
in the task guide. Single-image inference needs a checkpoint and input image,
not the full split or training dataset.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
