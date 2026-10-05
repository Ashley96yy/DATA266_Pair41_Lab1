# Task 2 Results

Three from-scratch Yelp Polarity models were trained: a mean-embedding baseline, a CNN, and a BiGRU.

## Model comparison

                  model  accuracy  f1_macro  roc_auc  pr_auc    mcc  brier_score  ece_10_bins
baseline_mean_embedding    0.8914    0.8914   0.9574  0.9574 0.7835       0.0825       0.0530
       experimental_cnn    0.9054    0.9054   0.9660  0.9652 0.8108       0.0717       0.0301
     experimental_bigru    0.8880    0.8880   0.9601  0.9608 0.7763       0.0904       0.0711

Complete metrics are stored in `metrics_report.csv`. Checkpoints are stored in `checkpoints/`.
