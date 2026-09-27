# Task 2 - Yuyao Ding - data_processed

[Download data_processed from Google Drive](https://drive.google.com/drive/folders/1jzNHg-lIQ1ok9_8g7d_9rF2HsN9JOaO_?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task2_sentiment/Yuyao_Ding/data_processed/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`data_processed/data_processed/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

The current formal session expects this layout:

```text
task2_formal_20260927T020054Z_ba87cd57/
  preprocessing_config.json
  split_ids.json
  vocab.json
```

These files record the formal preprocessing, data split and vocabulary. Older
files directly under `data_processed/` belong to earlier work; do not substitute
them for this session's files. If large processed arrays or parquet files are
absent, regenerate them using the task guide and original Yelp Polarity data.
The saved-model demo uses the vocabulary/preprocessing embedded in each checkpoint
and does not need the external processed arrays.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
