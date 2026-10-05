# Task 2 — Model Metrics Report

This report consolidates the recorded test-set results for the three Task 2 sentiment models. The values come from `task2_submission_files/metrics_report.csv` and refer to the same comparison run.

## Model comparison

| Model | Accuracy | Precision<br>macro | Recall<br>macro | F1<br>macro | ROC-AUC | PR-AUC | MCC | Brier | ECE | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean-Embedding Baseline | 0.8946 | 0.8946 | 0.8946 | 0.8946 | 0.9574 | 0.9574 | 0.7893 | 0.0825 | 0.0530 | 0.5300 |
| CNN | 0.9062 | 0.9062 | 0.9062 | 0.9062 | 0.9660 | 0.9654 | 0.8131 | 0.0723 | 0.0342 | 0.5600 |
| BiGRU | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9651 | 0.9651 | 0.8001 | 0.0796 | 0.0562 | 0.6400 |

Lower Brier score and ECE indicate better probability calibration. Higher values are preferred for the remaining predictive metrics.

## Additional metrics and resources

| Model | F1<br>micro | F1<br>weighted | Parameters | Training<br>seconds | Examples/second | Peak memory<br>MB |
|---|---|---|---|---|---|---|
| Mean-Embedding Baseline | 0.8946 | 0.8946 | 3,840,129 | 2.2569 | 39877.3758 | PENDING |
| CNN | 0.9062 | 0.9062 | 5,107,969 | 3.6177 | 24877.8701 | 245.0947 |
| BiGRU | 0.9000 | 0.9000 | 5,022,977 | 3.1367 | 28692.8116 | 652.9653 |

## Recorded findings

- CNN has the strongest recorded predictive results: accuracy **0.9062**, macro-F1 **0.9062**, ROC-AUC **0.9660**, PR-AUC **0.9654**, and MCC **0.8131**.
- BiGRU is second on accuracy and MCC, while providing higher throughput than CNN in the recorded run.
- The mean-embedding baseline has the smallest parameter count and the highest recorded throughput, making it the lightest reference model.
- CNN has the lowest Brier score and ECE in this comparison, indicating the best recorded calibration among the three models.

## Statistical exports

The supplied all-model metrics file does not contain confusion matrices, bootstrap confidence intervals, or paired McNemar p-values. Those fields remain marked as pending in the source report and should be populated only after the corresponding notebook exports are generated.

| Analysis | Status |
|---|---|
| Confusion matrix for each model | PENDING |
| Accuracy 95% confidence interval | PENDING |
| Macro-F1 95% confidence interval | PENDING |
| MCC 95% confidence interval | PENDING |
| Baseline vs CNN McNemar test | PENDING |
| Baseline vs BiGRU McNemar test | PENDING |

## Reproducibility note

The comparison table uses the recorded all-model run. A separate later CPU-only CNN run reported different values and is not mixed into this comparison.
