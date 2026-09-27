# Task 2 - Yuyao Ding - checkpoints

[Download checkpoints from Google Drive](https://drive.google.com/drive/folders/18AzMJ8NbMmE7l38yGKZziBppHmq4LlIZ?usp=sharing)

The artifact files are stored in Google Drive and are not included in a Git clone.
This README is tracked so the download location is visible in the repository.
Use an account with read access; request access from the folder owner if needed.

## Restore location

Download/extract the **contents** of the Drive folder into this repository-relative
directory:

```text
task2_sentiment/Yuyao_Ding/checkpoints/
```

Keep filenames and nested run directories unchanged. Avoid an extra
`checkpoints/checkpoints/` directory after extraction. Preserve JSON file bytes
and the original model weights so recorded checksums remain valid.

## Required layout and use

Preserve the formal session and model subdirectories:

```text
task2_formal_20260927T020054Z_ba87cd57/
  baseline/  (best.pt, best_epoch_005.pt, last.pt)
  textcnn/   (best.pt, best_epoch_003.pt, last.pt)
  bilstm/    (best.pt, best_epoch_003.pt, last.pt)
```

The demo and reported test metrics use each model's `best.pt`: epoch 5 for the
baseline and epoch 3 for TextCNN/BiLSTM. All three `last.pt` files are from epoch 5.
Keep the numbered best files with the last states so their recorded references
continue to resolve. The demo also uses the saved source/evaluation records
already tracked in this repository.

See the [task setup, data and demo guide](../README.md) for commands and raw-data
sources. Original logs and manifests remain in the repository under
`reproducibility/`; downloading artifacts does not require retraining.
