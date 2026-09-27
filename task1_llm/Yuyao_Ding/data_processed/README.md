# Task 1 - Yuyao Ding - data_processed

[Download data_processed from Google Drive](https://drive.google.com/drive/folders/1EfWk8oRTRdwjsv6qLH9zMbljhpX-Dmqz?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task1_llm/Yuyao_Ding/data_processed/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`data_processed/data_processed/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

Expected files at this directory's top level:

- `vocabulary.json`: the matching character vocabulary; required by `src/demo.py`.
- `split_story_hashes.json`: the recorded training/validation story selection.
- `train_ids.npy` and `valid_ids.npy`: encoded arrays for training/evaluation.

The saved-model demo needs the vocabulary and checkpoint, but not the arrays.
If arrays are absent, restore the original TinyStories data and follow the
preprocessing instructions in the task guide. Keep the recorded vocabulary and
split files unchanged; the current model uses their saved checksums.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
