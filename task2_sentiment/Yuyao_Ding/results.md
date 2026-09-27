# Task 2 - Yuyao Ding

Status: formal training, evaluation and case analysis are complete for Yuyao's three models.
The main session is `task2_formal_20260927T020054Z_ba87cd57`. The executed notebook is retained unchanged.
All models were trained from scratch on the same cleaned Yelp Polarity training split.

| Model | Test accuracy | Macro-F1 | Best epoch | Training wall time |
| --- | --- | --- | --- | --- |
| Baseline | 93.24% | 0.9324 | 5 | 1.91 min |
| TextCNN | 93.88% | 0.9388 | 3 | 2.30 min |
| BiLSTM | 94.75% | 0.9475 | 3 | 10.16 min |

BiLSTM has the highest test accuracy and macro-F1 in this run. Its accuracy is
1.51 percentage points above the baseline; TextCNN improves by
0.64 points. TextCNN has the lowest ECE. These results compare one
configuration and one seed per model, not every possible version of each architecture.

## Models and shared settings

| Model | Architecture | Dropout | Purpose |
| --- | --- | --- | --- |
| Baseline | Trainable embeddings, masked mean pooling, one binary logit | 0.30 | Simple reference without word order |
| TextCNN | Trainable embeddings, 128 filters for each width 3/4/5, masked max pooling | 0.50 | Local sentiment phrases |
| BiLSTM | Trainable embeddings, one bidirectional layer, 128 hidden units per direction | 0.40 | Sequential context |

Shared settings: embedding dimension 128, vocabulary 50,000 including PAD/UNK,
maximum input length 192, Adam learning rate 0.001, batch size 256, seed 640,
BCEWithLogitsLoss, maximum five epochs and validation-loss patience two.
BiLSTM clips the gradient norm at 1. All three use the same epoch shuffles.
No pretrained embeddings or language models were used. Architecture, dropout and
gradient clipping differ, so this is not a controlled architecture-only ablation.
See the saved [configurations](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/configs.json).

## Data and preprocessing

The original files contain 560,000 training reviews and 38,000 test reviews, with
balanced labels 0 (negative) and 1 (positive). They were downloaded from the shared
Drive. The cleaning audit removed four conflicting-label training rows, 62 additional
duplicate training rows and seven training rows overlapping the official test text.
No invalid training rows were found. Text identity uses lowercasing and normalized
whitespace; this does not detect all semantic or near duplicates.

The remaining 559,927 training rows were split 90/10 with stratification and seed 640.
The official test set stayed intact. Vocabulary and length-slice bounds use training
data only. All models share these exact splits.

| Split | Rows | Negative | Positive | Empty processed | Over 192 tokens |
| --- | --- | --- | --- | --- | --- |
| train | 503934 | 251959 | 251975 | 35 | 23774 |
| val | 55993 | 27996 | 27997 | 4 | 2710 |
| test | 38000 | 19000 | 19000 | 1 | 1751 |

Text is lowercased; curly apostrophes and selected negation contractions are normalized.
Punctuation and stopwords are removed, preserving the configured negation/contrast words.
There is no stemming or lemmatization. The vocabulary keeps words occurring at least
twice, up to 50,000 entries. Inputs retain the first 192 processed tokens and use PAD;
empty processed text receives UNK instead of being dropped.

The case review exposed information loss in this preprocessing: 'less' and 'least'
can disappear, number-word rating cues such as 'four stars' can lose the number,
and literal backslash-n sequences can leave stray n tokens. The completed experiment
retains this original preprocessing. Improvements below are proposals for a separate
validation experiment, not changes silently applied to the reported model.

Raw-file SHA-256:

```text
train-00000-of-00001.parquet  87bbf1eee0dc21f1a2790f31d8e9fecb66d09b5ea54e8f246779597170c7c89b
test-00000-of-00001.parquet   43f89639c05b9a91634c8c972979e074942f59df15f6369ccb81a7f402af1a25
```

The [preprocessing manifest](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/preprocessing.json)
records the split counts, processed-file hashes and fingerprint `cddf497d844b2500f4bf5cfeceea5bbc91b0b3110d397d53a942f84a8dbdae0b`.

## Learning curves and checkpoint selection

| Epoch | Baseline train / val loss | TextCNN train / val loss | BiLSTM train / val loss |
| --- | --- | --- | --- |
| 1 | 0.2986 / 0.2024 | 0.2753 / 0.1773 | 0.2153 / 0.1618 |
| 2 | 0.2023 / 0.1887 | 0.1885 / 0.1641 | 0.1382 / 0.1426 |
| 3 | 0.1889 / 0.1844 | 0.1635 / 0.1553 | 0.1110 / 0.1382 |
| 4 | 0.1817 / 0.1826 | 0.1451 / 0.1586 | 0.0895 / 0.1511 |
| 5 | 0.1767 / 0.1824 | 0.1305 / 0.1591 | 0.0693 / 0.1503 |

The baseline's validation loss is nearly flat by epoch five. TextCNN and BiLSTM
reach their lowest validation loss at epoch three; later training lowers training
loss but raises validation loss. Both reach patience two at epoch five.
Evaluation reloads the best validation checkpoint, rather than the final weights.
This supports stopping the current runs without adding epochs simply to train longer.
No nonfinite training events or failed runs were recorded.

![Baseline loss](outputs/task2_formal_20260927T020054Z_ba87cd57/baseline/loss_curve.png)
![TextCNN loss](outputs/task2_formal_20260927T020054Z_ba87cd57/textcnn/loss_curve.png)
![BiLSTM loss](outputs/task2_formal_20260927T020054Z_ba87cd57/bilstm/loss_curve.png)

Accuracy curves and per-epoch values are saved alongside each loss plot.

## Test metrics and uncertainty

All metrics below use the same 38,000 test reviews and a fixed probability threshold
of 0.5. Values are fractions except MCC. Full-precision values and evidence paths are
in [metrics_report.csv](metrics_report.csv).

| Test metric | Baseline | TextCNN | BiLSTM |
| --- | --- | --- | --- |
| Accuracy | 0.932447 | 0.938842 | 0.947526 |
| Precision, macro | 0.932461 | 0.938904 | 0.947553 |
| Recall, macro | 0.932447 | 0.938842 | 0.947526 |
| F1, macro | 0.932447 | 0.938840 | 0.947526 |
| Precision, micro | 0.932447 | 0.938842 | 0.947526 |
| Recall, micro | 0.932447 | 0.938842 | 0.947526 |
| F1, micro | 0.932447 | 0.938842 | 0.947526 |
| Precision, weighted | 0.932461 | 0.938904 | 0.947553 |
| Recall, weighted | 0.932447 | 0.938842 | 0.947526 |
| F1, weighted | 0.932447 | 0.938840 | 0.947526 |
| ROC-AUC | 0.979742 | 0.984943 | 0.988794 |
| PR-AUC (trapezoidal) | 0.979582 | 0.985341 | 0.989155 |
| Average precision | 0.979538 | 0.985341 | 0.989155 |
| MCC | 0.864908 | 0.877746 | 0.895079 |
| Brier score (lower is better) | 0.050910 | 0.045303 | 0.039617 |
| ECE (lower is better) | 0.005033 | 0.002623 | 0.009108 |

| 95% bootstrap interval | Baseline | TextCNN | BiLSTM |
| --- | --- | --- | --- |
| Accuracy | 0.929947 - 0.935159 | 0.936474 - 0.941369 | 0.945474 - 0.949921 |
| Macro-F1 | 0.929946 - 0.935156 | 0.936473 - 0.941360 | 0.945472 - 0.949918 |
| MCC | 0.859910 - 0.870349 | 0.873040 - 0.882777 | 0.890981 - 0.899865 |

Intervals use 1,000 paired row bootstrap resamples, seed 640, and percentile bounds
2.5%/97.5%. They describe test-row sampling uncertainty conditional on these trained
models, not variability across retraining seeds or reviewer/business clusters.

| Model | True negative | False positive | False negative | True positive |
| --- | --- | --- | --- | --- |
| Baseline | 17769 | 1231 | 1336 | 17664 |
| TextCNN | 17725 | 1275 | 1049 | 17951 |
| BiLSTM | 18076 | 924 | 1070 | 17930 |

Accuracy and macro-F1 are close because the classes are balanced and the per-class
performance is similar. BiLSTM's higher ECE (0.0091)
than TextCNN's (0.0026) shows that classification
quality and probability calibration need not have the same ranking.

### Metric definitions

For each class, `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, and F1 is their harmonic
mean. Macro averages the two class scores; weighted averages by class support; micro
aggregates counts across both labels before computing the score. Accuracy is correct/n.
`MCC = (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`.

ROC-AUC integrates TPR against FPR. PR-AUC is trapezoidal area under precision versus
recall; average precision is the separately reported non-interpolated summary.
Brier = mean((P(positive)-label)^2). ECE is the weighted absolute accuracy-confidence
gap in ten equal-width bins, where confidence is the probability of the predicted class.
The frozen implementation is in the [training notebook snapshot](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/source_notebook.ipynb).

### Paired comparisons

| Baseline compared with | Baseline only correct | Other only correct | Exact p | Holm p |
| --- | --- | --- | --- | --- |
| textcnn | 770 | 1013 | 9.48e-09 | 9.48e-09 |
| bilstm | 669 | 1242 | 1.07e-39 | 2.14e-39 |

Both experimental models outperform the baseline on this fixed test set, with
Holm-adjusted p-values below 0.05. The exact two-sided McNemar test uses discordant
pairs and Holm adjustment covers these two planned comparisons. IDs and labels are
aligned before testing. No TextCNN-versus-BiLSTM significance test is claimed.
These tests do not establish superiority across seeds or other datasets.

## Slice results

| Test slice | n (negative / positive) | Baseline F1 | TextCNN F1 | BiLSTM F1 |
| --- | --- | --- | --- | --- |
| length_short | 9590 (3837 / 5753) | 0.9294 | 0.9346 | 0.9457 |
| length_medium | 18952 (9397 / 9555) | 0.9338 | 0.9415 | 0.9493 |
| length_long | 9458 (5766 / 3692) | 0.9262 | 0.9318 | 0.9406 |
| has_negation | 27949 (16481 / 11468) | 0.9253 | 0.9350 | 0.9444 |
| no_negation | 10051 (2519 / 7532) | 0.9260 | 0.9236 | 0.9342 |
| truncated | 1751 (1149 / 602) | 0.9038 | 0.9030 | 0.9057 |
| not_truncated | 36249 (17851 / 18398) | 0.9333 | 0.9401 | 0.9491 |
| empty_after_preprocessing | 1 (1 / 0) | 0.0000 | 0.0000 | 0.0000 |

These are macro-F1 scores within each slice. Raw word-count boundaries 51 and 173
come from training quartiles; the saved source defines their boundary handling.
Negation is detected from raw text by the fixed rule. Truncation means more than
192 processed tokens. Slices overlap and have different class balances, so their
differences are descriptive, not causal comparisons.

Truncated reviews are harder for all three models. BiLSTM's error rate is 8.45%
there versus 5.09% on untruncated reviews. This does not prove that truncation alone
caused each error. The one empty processed test review is negative and is missed by
all three models; this is a single case, not a reliable subgroup estimate.

## Training resources

| Measure | Baseline | TextCNN | BiLSTM |
| --- | --- | --- | --- |
| Trainable parameters | 6,400,129 | 6,597,377 | 6,664,449 |
| Epochs completed | 5 | 5 | 5 |
| Training loop, seconds | 105.48 | 127.77 | 577.64 |
| Training wall time, seconds | 114.32 | 138.06 | 609.81 |
| Training examples / second | 23888.54 | 19720.45 | 4362.04 |
| Peak CUDA allocation, MiB | 166.98 | 298.79 | 378.29 |
| Process-lifetime peak RSS, GiB | 6.45 | 6.45 | 6.45 |

The machine used an Intel Core i9-14900KF and RTX 4090 on Windows 11, with Python
3.12.14, PyTorch 2.8.0+cu128 and cuDNN 91002. Package versions are preserved in the
[environment manifest](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/environment.json).
Training wall time includes validation and checkpoint work in each model loop, but
excludes preprocessing and final test evaluation. Throughput uses actual training
examples divided by training-loop time. CUDA memory is peak allocated tensor memory,
not total device memory. RSS is a process-lifetime peak shared by sequential runs;
it is not an isolated per-model RAM benchmark. There were no repeated timing trials.

## Error analysis and interpretation

The [case analysis](failure_analysis.md) covers 20 distinct errors per model:
five confident false positives, five confident false negatives, five closest to
the decision threshold, and five from the predefined long/negation slices.
There are 60 model-case entries and 53 unique reviews, with no category shortages.
The selected cases illustrate failures; they are not a random sample of all errors.

The baseline loses phrase scope and target information, with especially clear
examples where stopword removal changes the meaning. CNN and BiLSTM improve overall
classification but still confuse past versus current experience, a chain versus one
location, and complaints followed by a successful resolution. Some supplied labels
are difficult to infer from the text. These are flagged as ambiguous, not relabeled.

Useful follow-ups are a lighter stopword rule, normalization of literal escape
sequences and informal contractions, and a controlled comparison of head-only versus
head-plus-tail inputs. Keep each change separate and select settings on validation
data. Test-case observations motivate hypotheses; the current test results must not
be reused as an unbiased final estimate after tuning against these same cases.

## Reproduction, evidence and remaining team work

The [run guide](README.md) gives environment setup, smoke/formal runs, resume and
the independent [checkpoint demo](src/demo.py). The demo loads the saved vocabulary,
preprocessing and exact model definitions without reading raw data or training.
The original executed notebook, raw logs, checkpoints, metrics and run manifests
remain unchanged by this report work. Only the three analysis columns in each error
review CSV and its review-status file were completed.

Each model's run ID is `task2_formal_20260927T020054Z_ba87cd57_MODEL`. Evidence:

- **Baseline**: [manifest](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/baseline.json), [raw log](../../reproducibility/raw_logs/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57_baseline.jsonl), [evaluation](outputs/task2_formal_20260927T020054Z_ba87cd57/baseline/evaluation.json), [checkpoint](checkpoints/task2_formal_20260927T020054Z_ba87cd57/baseline/best.pt). SHA-256 `751be42e542837cb647464d4ab0e6bff48013578b3b7ff14228d0fc5ddd8c373`.
- **TextCNN**: [manifest](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/textcnn.json), [raw log](../../reproducibility/raw_logs/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57_textcnn.jsonl), [evaluation](outputs/task2_formal_20260927T020054Z_ba87cd57/textcnn/evaluation.json), [checkpoint](checkpoints/task2_formal_20260927T020054Z_ba87cd57/textcnn/best.pt). SHA-256 `70f90b9299dee6d429b32341b09bf877fad946806f3925771ad25ee6b142dec2`.
- **BiLSTM**: [manifest](../../reproducibility/manifests/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57/bilstm.json), [raw log](../../reproducibility/raw_logs/Yuyao_Ding/task2_formal_20260927T020054Z_ba87cd57_bilstm.jsonl), [evaluation](outputs/task2_formal_20260927T020054Z_ba87cd57/bilstm/evaluation.json), [checkpoint](checkpoints/task2_formal_20260927T020054Z_ba87cd57/bilstm/best.pt). SHA-256 `566af5cbdacacec8e564f8f1c97e573fa9901f52e54f96e8dde7fb88dc6b0a7b`.

Windows CPU/CUDA pipeline checks and formal CUDA training passed. The separate demo
and prediction checks are recorded in the [closeout verification](../../reproducibility/manifests/Yuyao_Ding/task2_closeout_checks.json).
Mac/Linux/Colab portability is implemented and documented, but these operating
systems have not been physically tested here. Cross-device bitwise identity is not promised.

The teammate's verified final architectures and results are still needed for the
required team comparison and joint discussion. The final combined PDF, off-machine
checkpoint backup/download link, and destination-machine check remain team handoff work.
The model choices above describe Yuyao's work only.
