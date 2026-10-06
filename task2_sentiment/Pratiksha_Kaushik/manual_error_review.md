# Task 2 - Manual Review of 20 Model Errors (Pratiksha Kaushik)

**Model reviewed:** `experimental_cnn` (CNN, kernels 3/5/7). It is my best model: test accuracy 0.9054, macro-F1 0.9054.
**Decision threshold:** 0.56 (tuned on validation). A review is predicted positive when P(positive) ≥ 0.56.
**Test set:** 5,000 Yelp Polarity reviews. The CNN made 473 errors (241 false positives, 232 false negatives).
**Source data:** `outputs/manual_error_review_all_models.csv` (same 20 rows, with the full review text).

## How the 20 errors were picked

The notebook (section 2.2.12) picked four groups of five errors, with no review used twice:

| Group | Rule | Rows |
|---|---|---|
| Confident false positives | true label negative, predicted positive, sorted by distance from the threshold (largest first) | 1-5 |
| Confident false negatives | true label positive, predicted negative, same sorting | 6-10 |
| Near-threshold errors | any wrong prediction, sorted by distance from the threshold (smallest first) | 11-15 |
| Slice-specific failures | wrong predictions on long reviews (> 100 tokens) or reviews with negation (not / never / no / n't), most confident first | 16-20 |

I then read every review in full and assigned the error type and the fix myself. The automatic group is only how each review was selected; the error type is my reading of why the model failed.

**Token counts** below are after my preprocessing (lowercase, punctuation removed, 47 stopwords removed). The model only sees the first 180 tokens.

## Error types used

| Code | Error type | Meaning |
|---|---|---|
| MIX | Mixed sentiment | The review has both praise and complaints, and the model weighted the wrong part |
| CON | Lost contrast word | The turn in the review is marked by "but" / "however", and "but" is in my stopword list |
| CMP | Comparative / superlative negative | Negative meaning comes from a phrase like "least favorite", not from a negative word |
| NEG | Negation cue misread | A negation word changes the meaning, or a negation word appears without being negative |
| END | Late sentiment shift | The final verdict comes at the end of the review and goes against most of the text |
| SAR | Sarcasm / indirect tone | Sentiment is implied, with few normal sentiment words |
| ENT | Wrong target | Sentiment is about a different business or event than the one being reviewed |
| OOD | Out-of-distribution text | Text the model can't read well (another language) |
| HDG | Hedged / mild sentiment | Weak sentiment, so the probability lands near the threshold |
| LBL | Possible label noise | The text doesn't clearly support its own label |
| BIAS | Topic-word bias | Neutral topic words (for example "customer service", "cancel") pull towards one class |

---

## A. Confident false positives (true: negative, predicted: positive)

### Error 1

> "of all the buffets I had in Vegas this was my least favorite!"

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.9999 | short | no | 6 |

**Error type:** CMP (comparative / superlative negative)

**What went wrong:** after preprocessing, the model sees only `all buffets had vegas least favorite`. "favorite" is one of the strongest positive words in Yelp reviews. "least" flips it, but "least favorite" is rare compared with "favorite" on its own, so the CNN's 3-gram filters fire on the positive word. The review is only 6 tokens long, so nothing else can outweigh it.

**Testable fix:** build a check set of about 50 test reviews containing "least favorite", "least impressive" or "worst of all". Compare the error rate on this set with the CNN as-is and with a CNN that gets an extra bigram feature (`least_favorite`, `least_impressive`). Success means the error rate on the check set drops with no loss in overall macro-F1.

### Error 2

> "Dr.Adams is great! Rest of office parley run. Long wait time, violations of patient privacy, loud staff!"

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.9998 | short | no | 15 |

**Error type:** MIX (mixed sentiment)

**What went wrong:** "is great!" is a very strong positive phrase, and the CNN uses max pooling, so one strong positive feature can dominate. The complaints ("long wait time", "violations of patient privacy", "loud staff") are spread out and use words that aren't strongly negative on their own. The reviewer's overall verdict is negative (praise for one person, complaints about the rest), but the model has no idea which part matters more.

**Testable fix:** score each sentence separately with the same CNN and average the sentence probabilities instead of scoring the whole review at once. Measure the false-positive count on test reviews that have both a positive and a negative sentence. Success means fewer false positives on this subset.

### Error 3

> "Good location and friendly staff. But need add more classes for what they charge a month"

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.9997 | short | no | 10 |

**Error type:** CON (lost contrast word)

**What went wrong:** the first sentence is clearly positive ("good location", "friendly staff"). The complaint comes after "But", and **"but" is in my stopword list, so it is deleted** before the model sees the review. Without it, the complaint ("need add more classes for what they charge") is a neutral-sounding phrase, so the positive opening wins.

**Testable fix:** remove "but" (and keep "however") from the stopword list and retrain the CNN with everything else the same. Compare the false-positive count on test reviews that contain "but". Success means fewer false positives on that subset, and no drop in overall macro-F1.

### Error 4

> "By far, the least impressive Cirque du Soleil show I have seen. The performers are all amazing, but for some reason, it was not showcased or directed well. We spent over $130 on each ticket, and we had excellent seats, which were very comfortable, But the show seem to drag on. The sand artist was really cool and different."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.9996 | short | yes | 37 |

**Error type:** MIX (mixed sentiment), with CMP and CON

**What went wrong:** this review packs in many strong positive words: "amazing", "excellent seats", "very comfortable", "really cool". The negative verdict comes from "least impressive" (a superlative negative, as in Error 1), "not showcased or directed well" and "drag on", and it is split by two "but"s that the model never sees. Counting phrases, the positives win.

**Testable fix:** the same "keep but" experiment as Error 3. In addition, compare the CNN and the BiGRU only on test reviews containing "but": the BiGRU reads the review in order, so it should handle the turn better. If the BiGRU has a lower error rate on this subset, it supports combining the two models.

### Error 5

> "Attended the @SpaFitFinder launch party, with @spacephx. I can't say much about the place, other than, even taking into consideration the number of people present, it felt cramped. The TrimTini (?) I had was really delicious; vodkatini with lemon zest. A very nice gentleman, who works at Urban 7, was very gracious in making sure to stop by every time he brought out a new platter of hors d'oeuvres. Everything he brought was delicious, including the ceviche ... Urban 7 isn't the type of place I'm attracted to, so it's not likely I will return any time soon. But it was a nice event, and I enjoyed myself."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.9994 | medium | yes | 83 |

**Error type:** LBL (possible label noise), with MIX

**What went wrong:** most of the text is positive ("really delicious", "very nice gentleman", "very gracious", "everything he brought was delicious", "I enjoyed myself"). The only negatives are "felt cramped" and "not likely I will return". The label is negative (1-2 stars), probably because the reviewer is rating the venue, not the event. A human reading the text would also guess positive, so the model's mistake is understandable.

**Testable fix:** count how many test errors look like this. Take a random sample of 50 confident false positives and mark each one as "text supports its label" or "text doesn't support its label". If more than about 20% don't, part of the error rate is label noise that no model fix can remove, and that should be reported as a limit on accuracy.

---

## B. Confident false negatives (true: positive, predicted: negative)

### Error 6

> "Great gym, but if you ever try to cancel your agreement count on paying 2-3 months longer than expected. I've been trying to cancel my account because I moved across the country. They won't cancel it unless I send in a certified letter. I've done that and am still getting charged months later."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0010 | short | no | 38 |

**Error type:** MIX (mixed sentiment), with BIAS and LBL

**What went wrong:** the reviewer says "Great gym" once and then spends the whole review on a billing complaint ("cancel" three times, "still getting charged months later"). These are words the model has learned from many negative reviews. The label is positive, presumably because the reviewer still rated the gym itself well. From the text alone, negative is the more natural reading.

**Testable fix:** check the topic-word bias directly. Take test reviews containing "cancel" or "charged" and compare their false-negative rate with the overall false-negative rate. If it is much higher, try down-weighting these words (for example, train with them masked 50% of the time) and check whether the gap shrinks.

### Error 7

> "I had dinner here last weekend, and decided to order the gnocchi ... I thought it was either overcooked and or dry, so sent it back for another order ... The next order wasn't much better, and the waitress and manager apologized and comped me my dinner. Even though I didn't like my food, I still give them 3 1/2 stars for trying to make it right and apologizing. The other thing that was annoying was that we ordered bottled water ... The atmosphere is pleasant and my wife liked her mussels."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0011 | medium | yes | 97 |

**Error type:** MIX (mixed sentiment), with an explicit rating the model can't use

**What went wrong:** the story is mostly negative ("overcooked", "dry", "wasn't much better", "didn't like my food", "annoying"). The reviewer states the verdict directly ("I still give them 3 1/2 stars"), but my preprocessing removes punctuation and the model has no idea that "3 1/2 stars" is a rating. "pleasant" and "liked" at the end are too weak to balance the rest.

**Testable fix:** add a simple rule-based feature that finds "N stars" / "N 1/2 stars" in the text and passes the number to the classifier. Measure accuracy on test reviews that mention a star rating, with and without the feature. Success means a clear accuracy gain on that subset.

### Error 8

> "For all the people who sit down and don't get any "service," there are signs that clearly say to order at the bar. So I would start with that. Then do this in any order you'd like: -cold beer -wings -cheesecurds Your welcome."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0021 | short | no | 31 |

**Error type:** SAR (sarcasm / indirect tone)

**What went wrong:** the review is written as advice to other customers, not as praise. It contains "don't get any service" (a phrase that is almost always negative in Yelp), and the positive meaning is only implied: the recommended list of beer, wings and cheese curds. There are no normal positive words.

**Testable fix:** this is hard for any model that reads phrases. As a measurable step, count how many test reviews quote a complaint from other people ("people who ... don't get ...") and check the error rate on them. If it is high, it should be reported as a known weakness rather than fixed.

### Error 9

> "So instead of going to Gibson Automotive to get my wifes battery swapped out we had $20.00 off $100.00 purchase coupon for Pep Boys ... they said that we needed a new alternator and not only that the $20.00 coupon only works for in store purchases ... 5 minutes later Larry Gibson comes out with the tests from 2 different machines saying nothing is wrong with the alternator. No charge and hope to see you soon......Pep Boys wanted over $460.00 ... Scam Artists (Pep Boys) is more like it. I have been taking my cars to Gibson since they were at their old location on South. I do not trust any other mechanics in town...."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0023 | long | yes | 105 |

**Error type:** ENT (wrong target)

**What went wrong:** the review is about Gibson Automotive, and it is positive about Gibson. But most of the text describes a bad experience at a different shop (Pep Boys): "scam artists", "wanted over $460", "do not trust". The model has no idea which business is being reviewed, so it scores the overall tone, which is negative.

**Testable fix:** raise MAX_LEN from 180 to 400 to check that the end isn't being cut. If that doesn't change the prediction, test a sentence-level model that gives more weight to sentences mentioning the reviewed business. As a smaller check, measure the error rate on test reviews that name a competitor ("instead of", "unlike", "compared to").

### Error 10

> "Outstanding food quality and customer service by the waiter."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0028 | short | no | 6 |

**Error type:** BIAS (topic-word bias). The cause is not confirmed.

**What went wrong:** this is a clearly positive review, and the model is almost certain it is negative. I can't see the reason from the text. My best guess is that "customer service" appears far more often in complaints than in praise in my 18k training reviews, and "waiter" also shows up mostly in complaint stories. "outstanding" may be rare in the sample.

**Testable fix:** occlusion test. Remove one word at a time from this review and record how P(pos) changes. If removing "customer service" makes the probability jump, the bias is confirmed. Then check the training vocabulary count for "outstanding": if it is rare, training on more data (100k+ reviews) should fix this kind of error.

---

## C. Near-threshold errors (P(pos) within 0.01 of 0.56)

### Error 11

> "Pendant des années, je travaillais dans le coin et, pendant des années, j'y ai mangé leurs repas et sandwichs prêts-à-manger sur l'heure du lunch ... La qualité de leurs produits est toujours des plus élevées, l'accueil en restaurant et au comptoir est toujours à la hauteur et les prix sont quand même abordables pour le Vieux-Montréal. Bonnes salades, excellents sandwichs sur pain au levain maison, délicieux plats chauds pour emporter, excellente variété de boissons et superbe sélection de sucreries et chocolats locaux."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.5584 | long | no | 122 |

**Error type:** OOD (review written in French)

**What went wrong:** the review is in French. My vocabulary was built from English reviews, and accented characters are stored as escape codes in this dataset, so most tokens are unknown to the model. With almost no known words, the model has nothing to go on and outputs a probability close to 0.5. It missed the threshold by only 0.0016.

**Testable fix:** count non-English reviews in the test set (for example, reviews where more than 30% of tokens are unknown) and report their error rate separately. If it is close to 50%, either filter them out or mark the prediction as low-confidence. Success means the error rate on English reviews goes down once these are handled separately.

### Error 12

> "This is a refreshing change from your small quick dining options while in Vegas. You're usually either stuck with the run of the mill 24 hour hotel lobby diner, a subpar casual dining experience, mall food or gift shop snacks ... It's Vegas, everything is marked up, so get over it. 2 scoops of gelato will run you $8.08 ... and lunch specials ... start around $15, I believe."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.5536 | medium | no | 93 |

**Error type:** HDG (hedged / mild sentiment), with MIX

**What went wrong:** the reviewer praises the place by comparing it with bad alternatives, so many of the words are negative but describe *other* places ("stuck with", "run of the mill", "subpar"). The price comments ("marked up") add more negative words. The only clear positive is "refreshing change". The model ends up just under the threshold.

**Testable fix:** check calibration around the threshold. Apply temperature scaling (fit one temperature on the validation set) and see how many of the reviews within 0.05 of the threshold change class. Then tune a separate threshold for each length group and check whether near-threshold errors drop.

### Error 13

> "I was here on a random Friday afternoon. It was okay, a different experience than I usually have, but I probably won't come back any time soon ... The draft and bottle lists are unimpressive, but it'll get the job done ... They also ran out of grilled chicken, so we had to substitute chicken tenders. I'm wondering why, since it was SO packed (read: not at all), they ran out, but it happens. Redeeming points: Big Buck Hunter."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| negative | positive | 0.5668 | long | yes | 137 |

**Error type:** SAR (sarcasm), with HDG

**What went wrong:** the reviewer is mildly negative ("okay", "won't come back", "unimpressive", "ran out") and uses sarcasm ("SO packed (read: not at all)"). There are also neutral-to-positive words ("perfect balance of crisp and gooey", "redeeming points", "decent liquor selection"). The tone is lukewarm rather than negative, so the probability sits just above the threshold.

**Testable fix:** build a small check set of test reviews containing sarcasm markers ("(read:", "yeah right", "SO" in capitals followed by a negation). Compare the CNN's error rate on this set with its overall rate. If it is much higher, report sarcasm as a separate failure mode. A quick test is whether keeping capital letters (no lowercasing) for all-caps words helps.

### Error 14

> "The Exchange has the biggest inventory of used CDs and movies in the city, so even if one location doesn't have what you're looking for, another location might. I bought a bunch of CDs here before I started downloading everything, and I still occasionally purchase a PS2 game or DVD here. The prices could stand to be a couple bucks cheaper, but otherwise it's all good."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.5529 | medium | yes | 41 |

**Error type:** HDG (hedged / mild sentiment), with CON

**What went wrong:** the review is factual and mildly positive. The only evaluative parts are "biggest inventory", the price complaint, and "but otherwise it's all good". The "but" that turns the review back to positive is removed by my stopword list. "doesn't have" adds a negation cue, so the model ends up just under the threshold.

**Testable fix:** the same "keep but" retrain as Error 3. Also check whether this review moves above the threshold after temperature scaling (see Error 12). Success means near-threshold errors on reviews containing "but otherwise" go away.

### Error 15

> "Lovely place in the middle of nowhere. Bottom floor is the bar and the second floor is a game area. Appetizers were good and cold beer."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.5521 | short | no | 15 |

**Error type:** HDG (hedged / mild sentiment), with BIAS

**What went wrong:** "lovely" and "good" are positive, but "middle of nowhere" usually goes with complaints about location, and half the review is a neutral description of the floors. With only 15 tokens, a couple of words decide the result, and it lands just below the threshold.

**Testable fix:** short reviews are my CNN's weak slice (error rate 0.097). Test adding a kernel of width 2 for short phrases, and separately tune the threshold for reviews with 40 tokens or fewer on the validation set. Success means a lower error rate on the short slice.

---

## D. Slice-specific failures (long reviews or reviews with negation)

### Error 16

> "UPDATED. My initial very frustrated and dramatic review read as follows: Bililng practices are at best negligent and at worst fraudulent ... Lo and behold a mysterious charge from the Studio shows up for $130 ... I am now disputing the charge via my credit card. The yoga is great but all these terrible billing practices make it not worth your time in the long run. Updates: After reading this, the studio owner went out of her way to contact me to try and correct the error ... resolved the situation in a way that goes above and beyond ... I am thus updating my review to 4 stars ..."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0035 | long | yes | 200 |

**Error type:** END (late sentiment shift), made worse by truncation

**What went wrong:** the review quotes the reviewer's original angry review first ("fraudulent", "terrible billing practices", "not worth your time") and only then gives the positive update. The review has 200 tokens, but **my model reads only the first 180**. The cutoff falls right where the reviewer says "updating my review to 4 stars", so the clearest positive statement and everything after it ("they certainly get 5 for the way they managed this") are cut off. The model sees mostly the old negative review.

**Testable fix:** retrain with MAX_LEN = 400, or keep the first 90 and last 90 tokens instead of the first 180. Compare the error rate on the long slice (> 100 tokens; currently 0.092 for the CNN). Success means a lower error rate on long reviews, with this review classified correctly.

### Error 17

> "I expected this location to be similar to the one in PHX: underwhelming and overpriced. However!!! I experienced something much better. It just seems a bit more quant..put together..and a place that encourages interaction between guests and invites you to truly enjoy yourself ... I'm not totally sure. This place deserves the hype."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0048 | medium | yes | 48 |

**Error type:** NEG (negation / expectation cue misread), with CON

**What went wrong:** the first sentence contains strong negative words ("underwhelming", "overpriced"), but they describe the reviewer's *expectation*, which the rest of the review contradicts. "However!!!" marks the turn and is kept, but the CNN doesn't model what "expected ... however" means. "I'm not totally sure" adds a negation cue. The positive ending ("much better", "deserves the hype") isn't enough.

**Testable fix:** compare CNN vs BiGRU on test reviews containing "expected" + "however" / "but". The BiGRU reads in order and should handle this pattern better. If it does, test a simple average of the CNN and BiGRU probabilities and check that the negation-slice error rate drops (currently 0.096).

### Error 18

> "The food is crap. I'm not trying to be mean, but it really is horrible ... They were probably the worst nachos I've ever had ... We ended up just going back to her place and lying on the floor for a while because our stomachs were so upset. Only drink at Barney's. If you eat, stick to fries ... Obviously, Thursday night gets to keep the four star rating. I can't help it. I will never get too old for Barney's Thursday nights."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0072 | long | yes | 142 |

**Error type:** END (late sentiment shift)

**What went wrong:** almost the whole review is a strongly negative food complaint ("crap", "horrible", "worst nachos", "stomachs were so upset"). The rating is positive because the reviewer is really rating Thursday-night drinks, which they say only at the end ("Thursday night gets to keep the four star rating"). This review is 142 tokens, so it is **not** truncated. The model sees the ending, but the negative part is much larger.

**Testable fix:** give the last sentences more weight. Train a variant that adds a second input made of only the last 40 tokens, and combine the two scores. Compare error rates on test reviews whose last sentence has the opposite sentiment from the rest. As a simpler first check, the star-rating feature from Error 7 would also catch "four star rating".

### Error 19

> "A friend and I visited this theatre the other day and the movie we intended on seeing was no longer showing. It was cancelled because another movie was doing so well. I was surprised when they told me and my friend to come inside for free passes! The staff was extremely friendly and apologized several times for the movie being cancelled. I was very impressed with the customer service and will drive out of my way to come to this location again!"

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0101 | medium | yes | 43 |

**Error type:** MIX (mixed sentiment), with BIAS

**What went wrong:** the review starts with a negative event ("no longer showing", "cancelled" twice, "apologized"), and those words are common in complaints. The positive verdict is clear to a human ("extremely friendly", "very impressed", "will drive out of my way to come ... again"), but "customer service" again pulls towards negative, as in Error 10.

**Testable fix:** the same occlusion test as Error 10. Remove "customer service" and "cancelled" one at a time and record how P(pos) changes. If both make the probability rise sharply, these topic words are biasing the model. Then retrain on a larger sample (100k reviews) and check whether reviews containing "customer service" still have a higher false-negative rate than average.

### Error 20

> "It's hard to find a non corporate coffee outlet in Las Vegas, but I found this place on Flamingo. Good coffee, not just a conversation piece. I ordered the Mocha, much better that I would have expected from a drive in. No pastries offered here, but a good choice of hot and frozen coffees."

| True | Predicted | P(pos) | Length | Negation | Tokens |
|---|---|---|---|---|---|
| positive | negative | 0.0111 | short | yes | 34 |

**Error type:** NEG (negation cue misread)

**What went wrong:** the review has several negation and negative-looking words that aren't negative here: "hard to find", "non corporate", "not just a conversation piece", "No pastries offered". I kept "not" and "no" on purpose, because they usually flip sentiment, but in this review they don't attach to anything negative. The positives ("good coffee", "much better", "good choice") are short and get outweighed.

**Testable fix:** count negation words only when they come right before an adjective or sentiment word (for example, "not good" counts but "not just" doesn't). Retrain with a simple negation-scope rule that joins "not" + the next word into one token (`not_good`), and compare the negation-slice error rate (currently 0.096) and the overall macro-F1.

---

## Summary

| # | Group | True → Pred | P(pos) | Error type |
|---|---|---|---|---|
| 1 | confident FP | neg → pos | 0.9999 | CMP |
| 2 | confident FP | neg → pos | 0.9998 | MIX |
| 3 | confident FP | neg → pos | 0.9997 | CON |
| 4 | confident FP | neg → pos | 0.9996 | MIX (+CMP, CON) |
| 5 | confident FP | neg → pos | 0.9994 | LBL (+MIX) |
| 6 | confident FN | pos → neg | 0.0010 | MIX (+BIAS, LBL) |
| 7 | confident FN | pos → neg | 0.0011 | MIX (stated rating) |
| 8 | confident FN | pos → neg | 0.0021 | SAR |
| 9 | confident FN | pos → neg | 0.0023 | ENT |
| 10 | confident FN | pos → neg | 0.0028 | BIAS |
| 11 | near threshold | pos → neg | 0.5584 | OOD |
| 12 | near threshold | pos → neg | 0.5536 | HDG (+MIX) |
| 13 | near threshold | neg → pos | 0.5668 | SAR (+HDG) |
| 14 | near threshold | pos → neg | 0.5529 | HDG (+CON) |
| 15 | near threshold | pos → neg | 0.5521 | HDG (+BIAS) |
| 16 | slice (long, negation) | pos → neg | 0.0035 | END (truncated) |
| 17 | slice (medium, negation) | pos → neg | 0.0048 | NEG (+CON) |
| 18 | slice (long, negation) | pos → neg | 0.0072 | END |
| 19 | slice (medium, negation) | pos → neg | 0.0101 | MIX (+BIAS) |
| 20 | slice (short, negation) | pos → neg | 0.0111 | NEG |

**Counts by main error type:** MIX 5, HDG 3, END 2, NEG 2, SAR 2, CMP 1, CON 1, ENT 1, OOD 1, BIAS 1, LBL 1 (20 in total).

Of the 20 errors, 14 are false negatives (true positive, predicted negative) and 6 are false positives (rows 1-5 and 13). The confident-FN group is negative by definition, and the near-threshold and slice groups happened to pick mostly positive reviews. Across the whole test set, the CNN's errors are balanced (241 FP vs 232 FN).

## Main takeaways

1. **Mixed reviews are the biggest problem.** Max pooling lets one strong phrase decide the result. A sentence-level model, or combining the CNN with the BiGRU, is the most promising fix.
2. **My preprocessing causes some errors.** Removing "but" (Errors 3, 4, 14) and cutting reviews at 180 tokens (Error 16) are easy changes to test.
3. **Some errors aren't really the model's fault.** Errors 5 and 6 are hard for a human too, and Error 11 is in French.
4. **Topic words like "customer service" and "cancel" may be biasing the model** (Errors 6, 10, 19). The occlusion test in Error 10 would confirm or rule this out.

## Fixes ranked by expected impact and cost

| Fix | Errors it targets | Cost | How to measure it |
|---|---|---|---|
| Remove "but" from the stopword list | 3, 4, 14 (+17) | very low, one-line change | FP count on reviews containing "but" |
| MAX_LEN 180 → 400, or keep head + tail | 16 (+9, 18) | low | error rate on the long slice |
| Train on more data (100k+ reviews) | 10, 15, 19 | medium | overall macro-F1, short-slice error rate |
| Average CNN + BiGRU | 4, 17, 19 | low, both models already trained | negation-slice error rate, McNemar vs CNN alone |
| Temperature scaling + per-slice thresholds | 12, 14, 15 | low | number of errors within 0.05 of the threshold |
| Negation-scope tokens (`not_good`) | 17, 20 | medium | negation-slice error rate |
| Star-rating feature | 7, 18 | medium | accuracy on reviews that mention stars |
| Non-English filter | 11 | low | error rate after removing non-English reviews |
| Occlusion test for topic-word bias | 6, 10, 19 | low, analysis only | change in P(pos) when the word is removed |
