# Task 2 Results - Pratiksha Kaushik

Dataset: Yelp Polarity (binary). I sampled 20,000 training reviews (18,000 train / 2,000 validation, stratified) and 5,000 test reviews. Seed 9002. All three models learn their embeddings from scratch and were trained for 5 epochs on a Tesla T4 in Colab.

All numbers below are from the final run in `RUN_LOG.txt` (the second block, started 2026-10-05 01:19:44). The decision threshold for each model was picked on the validation set, not on test.

## Test set comparison

| Model | Threshold | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| Mean-embedding baseline | 0.53 | 0.8914 | 0.8914 | 0.9574 | 0.9574 | 0.7835 | 0.0825 | 0.0530 |
| CNN (kernels 3/5/7) | 0.56 | **0.9054** | **0.9054** | **0.9660** | **0.9652** | **0.8108** | **0.0717** | **0.0301** |
| BiGRU | 0.27 | 0.8880 | 0.8880 | 0.9601 | 0.9608 | 0.7763 | 0.0904 | 0.0711 |

The CNN is the best model on every metric. The BiGRU ranks better than the baseline on ROC-AUC/PR-AUC but is slightly worse at its chosen threshold.

See `metrics.md` for confidence intervals, McNemar tests, slices and resource use. Raw numbers are in `metrics_report.csv` and `resources_report.csv`.
