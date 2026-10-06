# Processed data - Task 2 (Pratiksha Kaushik)

My Yelp Polarity split and vocabulary, exactly as used by `src/task2_yelp_sentiment_all_models.ipynb` (seed 9002).

The notebook builds these in memory. I recreated them with the notebook's own preprocessing code (`src/rebuild_data_processed.py`) and checked them against the final run: the vocabulary's most common tokens and counts match the notebook output (`n` 26,126, `not` 16,335, `had` 14,438, ...), and all 5,000 test labels match `outputs/predictions_experimental_cnn.csv` row by row.

| File | What it is | SHA-256 |
|---|---|---|
| `split_indices.json` | row numbers in the original `fancyzhx/yelp_polarity` parquet files: 18,000 train, 2,000 validation (both from the train file), 5,000 test (from the test file) | `942f4a8b33582090fb5504d2fa212141a1c688828ea020973440c8007a16820d` |
| `vocab.json` | `stoi` for the 30,000-token vocabulary (`<pad>` = 0, `<unk>` = 1), built from the training split only | `e39f91c64dfa7870b8ef9f22e1fb17cdaa5f205243528652ad96d41956f87913` |
| `dataset_stats.json` | row counts, class balance, tokens per review, vocab size and settings | `752e4e27e39d3564aa993cda4bafa70aa7d0289d5f46a2c484b73b2b4a970ee8` |

## Summary

| | Train | Validation | Test |
|---|---|---|---|
| Reviews | 18,000 | 2,000 | 5,000 |
| Negative / positive | 9,000 / 9,000 | 1,000 / 1,000 | 2,500 / 2,500 |
| Tokens per review (mean / median / max) | 83.88 / 61 / 623 | 85.94 / 62 / 649 | 83.1 / 61 / 587 |

Cleaning dropped 26 of the 560,000 training reviews (empty after preprocessing) and none of the test reviews. The model input is the first 180 tokens of each review.

To rebuild (downloads the two Yelp parquet files from Hugging Face):

```bash
python src/rebuild_data_processed.py
```
