# Failure Analysis - Pratiksha Kaushik

I reviewed 20 test errors from the best model (CNN, threshold 0.56). They are in `outputs/manual_error_review_all_models.csv`, with my error type and a fix I could test for each row. There are 5 rows in each group:

- confident false positives (model very sure it is positive, label negative)
- confident false negatives
- near-threshold errors (probability close to 0.56)
- slice errors (long reviews or reviews with negation)

## What I found

**1. Mixed reviews (9 of 20).** This was the most common problem. The review praises one thing and complains about another, for example "Dr. Adams is great! ... long wait time, violations of patient privacy". The CNN takes the max over phrase features, so one strong positive phrase can outweigh the rest. It doesn't know which part matters more.

**2. The ending decides the label (2 of 20).** Two long reviews were very negative for most of the text and only turned positive at the end ("I am thus updating my review to 4 stars", "Thursday night gets to keep the four star rating"). In the first one, the update starts right at the 180-token cutoff, so the "4 stars" part is cut off. The second one is only 142 tokens, so it isn't truncated, but the negative part simply outweighs the last few lines. The model gave both less than 0.01 probability of being positive.

**3. Negation and contrast (3 of 20).** For example "I expected this location to be ... underwhelming and overpriced. However!!! I experienced something much better". I kept not/never/no in preprocessing, but "but" is in my stopword list, so the contrast is lost. Words like "non corporate" and "No pastries" also trigger the negation pattern even though they aren't negative.

**4. Sarcasm and indirect wording (2 of 20).** "it was SO packed (read: not at all)" and the instruction-style review ("order at the bar ... Your welcome") have almost no normal sentiment words.

**5. Label noise or very weak signal (2 of 20).** One review is in French, so most of its tokens are unknown to the vocabulary. One negative-labelled review reads as neutral to positive ("it was a nice event, and I enjoyed myself").

**6. Short, clearly positive reviews (2 of 20).** "Outstanding food quality and customer service by the waiter." was predicted negative with 0.003 probability. "Lovely place in the middle of nowhere..." was also predicted negative, though close to the threshold. I could not see why from the text. These need a vocab check to see whether the words were rare in my 18k training sample. "customer service" also shows up a lot in complaints, which may pull these reviews towards negative.

## Fixes I would test next

| Fix | Errors it targets | How to measure it |
|---|---|---|
| Remove "but" (and "however") from the stopword list | contrast / mixed reviews | false-positive count and the negation-slice F1 |
| Raise MAX_LEN to 400, or keep the head and tail of the review | long reviews where the ending matters | error rate on the long slice |
| Add early stopping on validation F1 | BiGRU overfitting after epoch 3 | BiGRU test F1 and ECE |
| Use more training data (100k+) or subword tokens | rare words, French text | UNK rate and error rate on short reviews |
| Calibrate with temperature scaling | near-threshold errors | ECE and the number of errors with probability within 0.1 of the threshold |
| Filter or tag non-English reviews | the French example | count of non-English reviews in the test set |

## Slice summary

From `outputs/robustness_slices_all_models.csv`, the CNN's highest error rates are on low-punctuation reviews (0.104), short reviews (0.097) and reviews with negation (0.096). These match what I saw by hand: short, plain reviews have few strong words, and negation and contrast are only partly handled.
