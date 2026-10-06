# Task 2 - Yelp Polarity Sentiment (Pratiksha Kaushik)

Three sentiment classifiers trained from scratch (no pretrained embeddings) on Yelp Polarity:

- **Baseline:** mean of learned word embeddings + linear layer
- **Experimental 1:** CNN with kernel sizes 3, 5 and 7 + global max pooling
- **Experimental 2:** bidirectional GRU + masked mean pooling

Best model: **CNN**, with test accuracy and macro-F1 of 0.9054 and MCC 0.8108. It is significantly better than the baseline (McNemar p = 0.0006).

## Folder layout

```
src/
  task2_yelp_sentiment_all_models.ipynb  final notebook, trains and evaluates all 3 models (use this one)
  yelp_polarity_mean_embedding.ipynb     earlier single-model run of the baseline (CPU)
  yelp_polarity_cnn.ipynb                earlier single-model run of the CNN (CPU)
configs/            hyperparameters used in the final notebook
outputs/            predictions, confusion matrices (csv + png), training plot, McNemar,
                    slices, training history, error review, preprocessing_summary.md
metrics_report.csv  all test metrics + bootstrap CIs
resources_report.csv parameters, train time, throughput, peak memory, device
RUN_LOG.txt         raw log written by the notebook (the second block is the final run)
reproducibility_manifest.json  hardware, versions, seed, checkpoint list, sha256 of every output
results.md          short summary table
metrics.md          full metrics write-up
failure_analysis.md manual review of 20 errors
checkpoints/        model .pt files (gitignored, download link below)
data_processed/     not used, data is loaded directly from Hugging Face
```

## How to run

1. Open `src/task2_yelp_sentiment_all_models.ipynb` in Google Colab with a GPU runtime (I used a T4).
2. Run all cells. The first cell installs `datasets`, `scikit-learn`, `seaborn` and `psutil`. Locally you can run `pip install -r requirements.txt` instead.
3. Outputs are written to `$TASK2_RUN_DIR/all_models/`. The default is `/content/sample_data/TASK2_RUN_DIR/all_models/`.
4. Copy `metrics_report.csv`, `resources_report.csv`, `RUN_LOG.txt` and `outputs/` back into this folder.

The full run takes under a minute of training on a T4. The CPU runs of the single-model notebooks took about 30 s for the baseline and about 12 min for the CNN.

## Comparison with teammate (Yuyao Ding)

| Model | Test accuracy | Macro-F1 | MCC |
|---|---|---|---|
| Mine - baseline / CNN / BiGRU | 0.8914 / **0.9054** / 0.8880 | 0.8914 / 0.9054 / 0.8880 | 0.7835 / 0.8108 / 0.7763 |
| Yuyao - baseline / TextCNN / BiLSTM | 0.9324 / 0.9388 / **0.9475** | 0.9324 / 0.9388 / 0.9475 | 0.8649 / 0.8777 / 0.8951 |

Yuyao's models are 3-6 points higher, mainly because they trained on the full 560k training set and tested on all 38k test reviews. I used an 18k / 5k sample. They also used early stopping, while I kept the last epoch. In both setups the CNN beats the mean baseline significantly. The recurrent model was my weakest (the BiGRU overfit on small data), but it was Yuyao's strongest (the BiLSTM, with lots of data and early stopping). More details are in `metrics.md`, section 8.

## Settings

Seed 9002 (from my student ID). 20,000 training reviews split 90/10 into train/validation (stratified), and 5,000 test reviews. Vocabulary of 30,000 tokens, max length 180, batch size 128, 5 epochs, AdamW (lr 1e-3, weight decay 1e-4), gradient clipping at 1.0. The decision threshold is tuned on validation macro-F1 over 0.20-0.80. Full details are in `configs/`.

Preprocessing: lowercase, remove punctuation, remove a small stopword list, no stemming. "not", "never" and "no" are kept on purpose because they flip sentiment.

## Note on reproducibility

`RUN_LOG.txt` contains two runs of the same notebook. The baseline and BiGRU results are the same in both. The CNN changed slightly (0.9096 vs 0.9054) because cuDNN convolutions are not deterministic on GPU. I report the second, final run everywhere. The CPU single-model notebooks gave similar numbers (baseline 0.8912, CNN 0.9026).

## Checkpoints

The notebook saves `baseline_mean_embedding.pt`, `experimental_cnn.pt` and `experimental_bigru.pt` to `checkpoints/`. These are gitignored. Download link: _add Google Drive link here_
