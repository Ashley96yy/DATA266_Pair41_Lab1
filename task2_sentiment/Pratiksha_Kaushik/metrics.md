# Task 2 Metrics - Pratiksha Kaushik

No pretrained embeddings or pretrained language models were used. Every embedding is randomly initialized and learned only from my 18k training reviews.

Source files: `metrics_report.csv`, `resources_report.csv` and everything in `outputs/`. Test set is 5,000 reviews (2,500 per class). All values come from the same final run.

## 1. Main metrics

Precision, recall and F1 are macro averaged. Micro and weighted F1 are the same as accuracy / macro-F1 here because the test set is exactly balanced.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE (10 bins) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.53 | 0.8914 | 0.8921 | 0.8914 | 0.8914 | 0.9574 | 0.9574 | 0.7835 | 0.0825 | 0.0530 |
| CNN | 0.56 | 0.9054 | 0.9054 | 0.9054 | 0.9054 | 0.9660 | 0.9652 | 0.8108 | 0.0717 | 0.0301 |
| BiGRU | 0.27 | 0.8880 | 0.8883 | 0.8880 | 0.8880 | 0.9601 | 0.9608 | 0.7763 | 0.0904 | 0.0711 |

## 2. 95% bootstrap confidence intervals (1,000 resamples of the test set)

| Model | Accuracy | Macro-F1 | MCC |
|---|---|---|---|
| Baseline | [0.8826, 0.8996] | [0.8826, 0.8996] | [0.7662, 0.7997] |
| CNN | [0.8974, 0.9136] | [0.8974, 0.9136] | [0.7950, 0.8273] |
| BiGRU | [0.8790, 0.8966] | [0.8788, 0.8965] | [0.7584, 0.7935] |

The CNN interval barely overlaps the baseline interval. The BiGRU and baseline intervals overlap almost completely.

## 3. Confusion matrices (rows = actual, columns = predicted)

| Model | TN | FP | FN | TP | Recall on negatives | Recall on positives |
|---|---|---|---|---|---|---|
| Baseline | 2281 | 219 | 324 | 2176 | 0.912 | 0.870 |
| CNN | 2259 | 241 | 232 | 2268 | 0.904 | 0.907 |
| BiGRU | 2183 | 317 | 243 | 2257 | 0.873 | 0.903 |

The baseline leans towards predicting negative (more false negatives). The CNN errors are almost evenly split. The BiGRU's low threshold (0.27) pushes it the other way, so it makes more false positives.

| Baseline | CNN | BiGRU |
|---|---|---|
| ![](outputs/confusion_matrix_baseline_mean_embedding.png) | ![](outputs/confusion_matrix_experimental_cnn.png) | ![](outputs/confusion_matrix_experimental_bigru.png) |

## 4. McNemar test (exact, paired, each model vs baseline)

| Comparison | Baseline right, other wrong | Baseline wrong, other right | p-value |
|---|---|---|---|
| Baseline vs CNN | 168 | 238 | 0.0006 |
| Baseline vs BiGRU | 275 | 258 | 0.488 |

The CNN improvement is statistically significant (p < 0.001). The BiGRU is not significantly different from the baseline.

## 5. Robustness slices (macro-F1 / error rate)

Length is counted in tokens after cleaning: short <= 40, medium 41-100, long > 100. Negation means the review contains not / never / no / n't. High punctuation means 3 or more `!` or `?`.

| Slice | n | Baseline | CNN | BiGRU |
|---|---|---|---|---|
| short | 1622 | 0.884 / 0.112 | 0.900 / 0.097 | 0.872 / 0.121 |
| medium | 1921 | 0.887 / 0.113 | 0.905 / 0.095 | 0.895 / 0.105 |
| long | 1457 | 0.897 / 0.099 | 0.905 / 0.092 | 0.884 / 0.112 |
| has negation | 2956 | 0.876 / 0.112 | 0.896 / 0.096 | 0.873 / 0.118 |
| no negation | 2044 | 0.879 / 0.103 | 0.891 / 0.092 | 0.874 / 0.104 |
| low punctuation | 3896 | 0.883 / 0.117 | 0.896 / 0.104 | 0.879 / 0.121 |
| high punctuation | 1104 | 0.922 / 0.078 | 0.940 / 0.060 | 0.919 / 0.081 |

The weakest slices for every model are short reviews, reviews with negation, and reviews with little punctuation. Reviews with lots of `!`/`?` are the easiest, probably because they are strongly worded. The CNN is best on every slice.

## 6. Resources (18,000 train rows x 5 epochs)

Hardware: all three models in this table were trained in the same Colab session on an **NVIDIA Tesla T4 GPU** (PyTorch 2.11.0+cu130). The earlier single-model runs in `src/yelp_polarity_mean_embedding.ipynb` and `src/yelp_polarity_cnn.ipynb` used a Colab CPU runtime (x86_64, PyTorch 2.11.0+cpu). Details are in `reproducibility_manifest.json`.

| Model | Parameters | Train time (s) | Examples/s | Peak GPU memory (MB) |
|---|---|---|---|---|
| Baseline | 3,840,129 | 6.35 | 14,181 | 100.5 |
| CNN | 5,107,969 | 18.98 | 4,742 | 236.1 |
| BiGRU | 5,022,977 | 16.03 | 5,616 | 558.2 |

Most parameters are in the embedding table (30,000 x 128 for the baseline, 30,000 x 160 for the others). The baseline is about 3x faster than the CNN and uses the least memory. The BiGRU uses more than twice the memory of the CNN.

## 7. Training curves (validation macro-F1 at threshold 0.5)

![Training loss and validation F1](outputs/training_loss_and_val_f1.png)

| Epoch | Baseline | CNN | BiGRU |
|---|---|---|---|
| 1 | 0.785 | 0.852 | 0.869 |
| 2 | 0.846 | 0.885 | 0.893 |
| 3 | 0.875 | 0.884 | **0.901** |
| 4 | 0.891 | 0.899 | 0.900 |
| 5 | **0.901** | **0.901** | 0.890 |

The BiGRU peaks at epoch 3 and then starts overfitting (train loss falls to 0.075 while validation F1 drops). I kept the last-epoch weights for all models, so the BiGRU result is probably a bit lower than it could be with early stopping. The baseline is still improving at epoch 5.

## 8. Comparison with my teammate (Yuyao Ding)

Yuyao's numbers are copied from their `task2_sentiment/Yuyao_Ding/metrics_report.csv` (run `task2_formal_20260927T020054Z_ba87cd57`). I did not rerun or change anything in their folder.

### Setup differences

| | Mine | Yuyao |
|---|---|---|
| Training data | 18,000 reviews (sample of 20k) | full 560k training set minus a validation split |
| Test set | 5,000 reviews (sample) | all 38,000 test reviews |
| Models | mean-embedding, CNN (3/5/7), BiGRU | mean-embedding, TextCNN (3/4/5), BiLSTM |
| Vocab / max length | 30,000 / 180 | 50,000 / 192 |
| Optimizer / batch | AdamW, batch 128 | Adam, batch 256 |
| Epoch used | last epoch (5) | best validation epoch (early stopping, patience 2) |
| Threshold | tuned on validation | fixed 0.5 |
| Hardware | Tesla T4 (Colab) | RTX 4090 |

### Test results

| Model | Accuracy | Macro-F1 | ROC-AUC | MCC | ECE | Params |
|---|---|---|---|---|---|---|
| Mine - baseline | 0.8914 | 0.8914 | 0.9574 | 0.7835 | 0.0530 | 3.84M |
| Mine - CNN | 0.9054 | 0.9054 | 0.9660 | 0.8108 | 0.0301 | 5.11M |
| Mine - BiGRU | 0.8880 | 0.8880 | 0.9601 | 0.7763 | 0.0711 | 5.02M |
| Yuyao - baseline | 0.9324 | 0.9324 | 0.9797 | 0.8649 | 0.0050 | 6.40M |
| Yuyao - TextCNN | 0.9388 | 0.9388 | 0.9849 | 0.8777 | 0.0026 | 6.60M |
| Yuyao - BiLSTM | 0.9475 | 0.9475 | 0.9888 | 0.8951 | 0.0091 | 6.66M |

McNemar vs baseline: my CNN p = 0.0006 and my BiGRU p = 0.49 (not significant). Yuyao's TextCNN p = 9.5e-9 and BiLSTM p = 1.1e-39 (both significant).

### What the comparison shows

- **Yuyao's models are 3-6 points better.** The biggest reason is data: they trained on about 28x more reviews. Even their simplest baseline (0.932) beats my best model (0.905).
- **The ranking of the sequence models is different.** For Yuyao, the recurrent model (BiLSTM) is the best. For me, the recurrent model (BiGRU) is the worst. My BiGRU overfit after epoch 3 on only 18k reviews, and I kept the epoch-5 weights. Yuyao used early stopping and had enough data for the LSTM to learn longer context. So with small data a CNN is the safer choice, and the RNN only pays off once there is more data.
- **In both cases a CNN beats the mean baseline significantly.** Local phrase features help no matter how much data is used.
- **Their models are much better calibrated** (ECE 0.003-0.009 vs 0.03-0.07). More data plus early stopping keeps the probabilities from becoming over-confident.
- **The test sets are different sizes.** My confidence intervals are about 3x wider (5k vs 38k test reviews), so small differences between my models are less certain.
- **Speed is hard to compare** because the GPUs are different (T4 vs RTX 4090). Per model, my training took 6-19 seconds and theirs took 105-578 seconds, mainly because they used about 28x more data.

### What I would take from their setup

Early stopping on validation, training on much more of the 560k reviews, and a slightly longer max length. These are the same fixes my own failure analysis points to.

## Observations

- **CNN (local phrases) helps.** Using kernel widths 3, 5 and 7 lets it pick up short phrases like "not good" or "least favorite" that a mean of word vectors loses. It is also the best calibrated model (lowest Brier and ECE).
- **BiGRU (sequence context) did not help in this setup.** It learns fastest in the first epochs but overfits by epoch 5. The 0.27 threshold also shows its probabilities are shifted, which explains its higher ECE.
- **Baseline is a strong cheap reference.** It is only 1.4 points behind the CNN in accuracy, with the fewest parameters and the fastest training.

## Limitations

- Only 20k of the 560k training reviews were used, with 5 epochs and one seed.
- Reviews are cut at 180 tokens, which drops the ending of long reviews (one of the reviewed errors in `failure_analysis.md` comes from this).
- "but" is in my stopword list, so contrast words are removed before the model sees them.
- No early stopping, so the BiGRU used its overfit last epoch.
- Yelp Polarity stores line breaks as the literal text `\n`. My tokenizer turns these into a stray `n` token, which is the most common token in the vocabulary. This should be cleaned before tokenizing.
- GPU runs are not fully deterministic: an earlier run in the same log (01:14) gave CNN accuracy 0.9096 with threshold 0.50. The baseline and BiGRU numbers were identical in both runs.
