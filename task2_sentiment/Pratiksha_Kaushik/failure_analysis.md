# Failure Analysis - Pratiksha Kaushik

I reviewed 20 test errors from the best model (CNN, threshold 0.56). They are in `outputs/manual_error_review_all_models.csv`, with my error type and a fix I could test for each row. The detailed write-up of each error is in [manual_error_review.md](manual_error_review.md). There are 5 rows in each group:

- confident false positives (model very sure it is positive, label negative)
- confident false negatives
- near-threshold errors (probability close to 0.56)
- slice errors (long reviews or reviews with negation)

## All 20 reviewed errors

Model: CNN, threshold 0.56. *P(pos)* is the model's probability that the review is positive. A detailed write-up of each error (full excerpt, explanation, how to test the fix) is in [manual_error_review.md](manual_error_review.md).

### Confident false positives (true negative, predicted positive)

| # | Review excerpt | True | Pred | P(pos) | Slice | Error type | Testable fix |
|---|---|---|---|---|---|---|---|
| 1 | "of all the buffets I had in Vegas this was my least favorite!" | neg | pos | 0.9999 | short | CMP: superlative negative ('least favorite') read as positive | bigram feature for 'least/worst X'; test on a 50-review 'least/worst' check set |
| 2 | "Dr.Adams is great! Rest of office parley run. Long wait time, violations of patient privacy, loud staff!" | neg | pos | 0.9998 | short | MIX: one strong positive phrase outweighs the complaints | sentence-level scoring averaged over sentences; compare FP on mixed reviews |
| 3 | "Good location and friendly staff. But need add more classes for what they charge a month" | neg | pos | 0.9997 | short | CON: contrast lost because 'but' is a stopword | keep 'but' in preprocessing and retrain; compare FP on reviews with 'but' |
| 4 | "By far, the least impressive Cirque du Soleil show I have seen. The performers are all amazing, but for some ..." | neg | pos | 0.9996 | short, negation | MIX: many positive words, negative verdict split by 'but' | keep 'but'; compare CNN vs BiGRU on reviews containing 'but' |
| 5 | "Attended the @SpaFitFinder launch party, with @spacephx. I can't say much about the place, other than, even ..." | neg | pos | 0.9994 | medium, negation | LBL: text reads positive, label negative | audit 50 confident FPs for label noise and report the share |

### Confident false negatives (true positive, predicted negative)

| # | Review excerpt | True | Pred | P(pos) | Slice | Error type | Testable fix |
|---|---|---|---|---|---|---|---|
| 6 | "Great gym, but if you ever try to cancel your agreement count on paying 2-3 months longer than expected. I've ..." | pos | neg | 0.0010 | short | MIX: 'Great gym' then billing complaint (cancel/charged) | check FN rate on reviews with 'cancel'/'charged'; try masking these words in training |
| 7 | "I had dinner here last weekend, and decided to order the gnocchi since I had it at Bouchon in Yountville. I ..." | pos | neg | 0.0011 | medium, negation | MIX: negative story, positive stated rating (3 1/2 stars) | add a star-rating feature; compare accuracy on reviews that mention stars |
| 8 | "For all the people who sit down and don't get any \""service,\"" there are signs that clearly say to order at ..." | pos | neg | 0.0021 | short | SAR: advice/indirect tone, quotes other people's complaint | measure error rate on reviews quoting complaints; report as known weakness |
| 9 | "So instead of going to Gibson Automotive to get my wifes battery swapped out we had $20.00 off $100.00 ..." | pos | neg | 0.0023 | long, negation | ENT: negative about a competitor, positive about the reviewed shop | MAX_LEN 400 + error rate on reviews naming a competitor ('instead of') |
| 10 | "Outstanding food quality and customer service by the waiter." | pos | neg | 0.0028 | short | BIAS: 'customer service'/'waiter' pull towards negative (unconfirmed) | occlusion test per word; check training frequency of 'outstanding' |

### Near-threshold errors (probability close to 0.56)

| # | Review excerpt | True | Pred | P(pos) | Slice | Error type | Testable fix |
|---|---|---|---|---|---|---|---|
| 11 | "Pendant des années, je travaillais dans le coin et, pendant des années, j'y ai mangé leurs repas et sandwichs ..." | pos | neg | 0.5584 | long | OOD: review written in French | flag reviews with >30% unknown tokens; report their error rate separately |
| 12 | "This is a refreshing change from your small quick dining options while in Vegas. You're usually either stuck ..." | pos | neg | 0.5536 | medium | HDG: praise by comparison with bad places + price comments | temperature scaling + per-length thresholds; count errors within 0.05 of threshold |
| 13 | "I was here on a random Friday afternoon. It was okay, a different experience than I usually have, but I ..." | neg | pos | 0.5668 | long, negation | SAR: lukewarm review with sarcasm ('SO packed (read: not at all)') | sarcasm-marker check set; test keeping capitals for all-caps words |
| 14 | "The Exchange has the biggest inventory of used CDs and movies in the city, so even if one location doesn't ..." | pos | neg | 0.5529 | medium | HDG: mild positive, 'but otherwise it's all good' loses its 'but' | keep 'but' + temperature scaling; recheck near-threshold errors |
| 15 | "Lovely place in the middle of nowhere. Bottom floor is the bar and the second floor is a game area. ..." | pos | neg | 0.5521 | short | HDG: very short, 'middle of nowhere' pulls negative | add kernel width 2 and a short-review threshold; compare short-slice error |

### Slice-specific failures (long reviews / reviews with negation)

| # | Review excerpt | True | Pred | P(pos) | Slice | Error type | Testable fix |
|---|---|---|---|---|---|---|---|
| 16 | "UPDATED. My initial very frustrated and dramatic review read as follows: Bililng practices are at best ..." | pos | neg | 0.0035 | long, negation | END: positive update after quoted angry review, cut at 180 tokens | MAX_LEN 400 or head+tail tokens; compare long-slice error rate |
| 17 | "I expected this location to be similar to the one in PHX: underwhelming and overpriced. However!!! I ..." | pos | neg | 0.0048 | medium, negation | NEG: negative words describe the expectation, not the experience | compare CNN vs BiGRU on 'expected ... however'; test CNN+BiGRU average |
| 18 | "The food is crap. I'm not trying to be mean, but it really is horrible. I'd rather eat one of those Tornado ..." | pos | neg | 0.0072 | long, negation | END: very negative body, positive rating in the last lines (not truncated) | extra input from the last 40 tokens; or star-rating feature |
| 19 | "A friend and I visited this theatre the other day and the movie we intended on seeing was no longer showing. ..." | pos | neg | 0.0101 | medium, negation | MIX: negative event (cancelled) but praised service | occlusion test on 'customer service'/'cancelled'; retrain on 100k reviews |
| 20 | "It's hard to find a non corporate coffee outlet in Las Vegas, but I found this place on Flamingo . Good ..." | pos | neg | 0.0111 | short, negation | NEG: 'non corporate', 'No pastries', 'not just' are not negative | negation-scope tokens (not_good); compare negation-slice error rate |

## What I found

| Error type | Count | Rows |
|---|---|---|
| MIX - mixed sentiment, one strong phrase wins | 5 | 2, 4, 6, 7, 19 |
| HDG - hedged / mild sentiment near the threshold | 3 | 12, 14, 15 |
| END - verdict comes at the end of the review | 2 | 16, 18 |
| NEG - negation cue misread | 2 | 17, 20 |
| SAR - sarcasm / indirect tone | 2 | 8, 13 |
| CMP - superlative negative ("least favorite") | 1 | 1 |
| CON - contrast lost because "but" is a stopword | 1 | 3 |
| ENT - sentiment about a different business | 1 | 9 |
| OOD - non-English review | 1 | 11 |
| BIAS - topic words ("customer service") pull negative | 1 | 10 |
| LBL - text doesn't support the label | 1 | 5 |

**1. Mixed and hedged reviews are the main problem (8 of 20).** The CNN takes the max over phrase features, so one strong phrase ("is great!", "excellent seats") can decide the whole review even when the reviewer's verdict is the opposite.

**2. Some errors come from my own preprocessing.** "but" is in my stopword list, so the turn in the review is lost (rows 3, 4, 14). The 180-token limit cuts the positive update in row 16.

**3. The end of the review matters, but the model doesn't weight it.** Rows 16 and 18 are negative for most of the text and give a positive rating at the end.

**4. Negation is double-edged.** Keeping not/no/never helps in general, but in rows 17 and 20 the negation words don't make the review negative ("non corporate", "No pastries", "expected ... underwhelming. However!!! ... much better").

**5. Some errors are hard for anyone.** Row 5 reads positive with a negative label, row 6 is mostly a complaint with a positive label, and row 11 is in French.

**6. Possible topic-word bias.** "customer service", "cancel" and "charged" appear far more often in complaints, which may explain rows 6, 10 and 19. This is not confirmed yet; the occlusion test in `manual_error_review.md` would check it.

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
