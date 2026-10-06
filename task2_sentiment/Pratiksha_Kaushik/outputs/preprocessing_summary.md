# Preprocessing summary

All of this comes from the printed output of `src/task2_yelp_sentiment_all_models.ipynb` (sections 2.1.1 - 2.1.5).

## Raw data (fancyzhx/yelp_polarity on Hugging Face)

| Split | Rows | Negative (0) | Positive (1) | Missing text | Missing label |
|---|---|---|---|---|---|
| train | 560,000 | 280,000 | 280,000 | 0 | 0 |
| test | 38,000 | 19,000 | 19,000 | 0 | 0 |

Review length in words (train): mean 133, median 97, 90th percentile 282, 95th 372, 99th 608, max 1052. The plot is `data_review_length_and_class_balance.png`.

## Cleaning steps

1. Fill missing text with "" and convert labels to numbers. Drop rows whose label isn't 0 or 1.
2. Lowercase, then replace every character that is not a-z, 0-9, a space or an apostrophe with a space.
3. Split into word tokens (contractions like "wouldn't" stay as one token).
4. Remove a 47-word stopword list. "not", "never" and "no" are kept.
5. Drop rows that end up with 0 tokens: 26 of the 560,000 training reviews and none of the test reviews.
6. Stemming is off (`USE_STEMMING = False`).

## Sampling and split (seed 9002)

| Split | Rows | Class balance |
|---|---|---|
| train | 18,000 | 50 / 50 |
| validation | 2,000 | 50 / 50 |
| test | 5,000 | 50 / 50 |

The train and validation rows are a stratified 20,000-row sample of the 560k training set, split 90/10. The test rows are a stratified 5,000-row sample of the 38k test set.

## Vocabulary and encoding

- Built from the 18,000 training rows only: 30,000 tokens, including `<pad>` and `<unk>`.
- Most common tokens: n, not, had, food, place, good, out, all, like, just, get, very, one, here, time.
- Sequences are cut at 180 tokens and padded per batch (minimum length 7, so the size-7 CNN kernel always fits).

**Issue found:** the most common token is `n`. Yelp Polarity stores newlines as the literal text `\n`. My regex removes the backslash, which leaves a stray `n` token for every line break. It is harmless noise, but it wastes sequence length on long reviews. Next time, replace `\\n` with a space before tokenizing.

The notebook rebuilds the splits in memory each run (the seed makes them identical). The exact split row numbers, vocabulary and stats are saved in `data_processed/`.
