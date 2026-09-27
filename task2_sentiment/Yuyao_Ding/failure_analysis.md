# Task 2 error analysis - Yuyao Ding

Status: case-by-case analysis completed for `task2_formal_20260927T020054Z_ba87cd57`.
Each model has 20 distinct wrong predictions, five in each required selection category.
Across models there are 60 entries and 53 unique reviews. No category is short of cases.
The supplied labels, predictions, probabilities, selection order and raw text are unchanged.

An excerpt below is copied from the original review. Each observation was checked
against the full raw text and processed input. Information removed by preprocessing
or the 192-token boundary is directly observable; explanations of the model's internal
decision remain hypotheses, without attribution experiments. Proposed fixes have not
been tested or used to alter these models. Selection favors confidence extremes,
borderline cases and specified slices, so category counts are not error prevalence estimates.

## Patterns worth discussing

- Baseline: phrase scope and stopword loss are clear in `test_4742`, `test_17640`
  and `test_29123`; words needed for the negative comparison or explicit rating disappear.
- TextCNN: local positive or negative phrases can refer to the wrong entity or time.
  `test_37771` criticizes one branch after praising the chain; `test_30793` praises
  new owners after describing the old owners' poor service.
- BiLSTM: higher overall accuracy still leaves mixed-aspect and temporal errors.
  Sequence order alone does not guarantee that the final judgment or target is understood.
- Shared cases: `test_22807` is an update over an old complaint. `test_20061` loses
  its positive resolution beyond token 192. In contrast, `test_30958` loses only the
  last 'thank'; its positive manager-resolution passage remains, so truncation is
  not a sufficient explanation.
- Some short texts do not clearly support their supplied labels, including
  `test_20681` and `test_29330`. They remain scored using the official labels.

The [slice results](results.md#slice-results) show that truncated reviews are harder,
but length slices and token truncation are different. A long raw review can still fit
after stopword removal, and the analysis below checks that distinction case by case.

## Baseline cases

Checkpoint: `task2_formal_20260927T020054Z_ba87cd57/baseline/best.pt`, epoch 5.

[Annotated CSV](outputs/task2_formal_20260927T020054Z_ba87cd57/baseline/error_review.csv) | [Selection and analysis status](outputs/task2_formal_20260927T020054Z_ba87cd57/baseline/error_review_status.json)

### 1. test_4742 - Confident false positive

Label negative; predicted positive; P(positive)=1. Processed tokens: 2; retained at most 192.

> less than accomodating or helpful

**Preprocessing: lost comparison.** The processed input is only 'accomodating helpful'. Removing 'less than' reverses the meaning available to the classifier.

Proposed validation check: Keep comparison words such as less and than; retrain with the same split and compare validation errors on comparative phrases.

### 2. test_17640 - Confident false positive

Label negative; predicted positive; P(positive)=1. Processed tokens: 3; retained at most 192.

> my least favorite

**Preprocessing: lost comparison.** The processed input is 'buffets vegas favorite'. The word 'least', which makes this a negative comparison, has been removed.

Proposed validation check: Preserve least and other comparison operators, then compare validation accuracy with the original stopword rule.

### 3. test_20681 - Confident false positive

Label negative; predicted positive; P(positive)=0.99999714. Processed tokens: 2; retained at most 192.

> Large variety

**Ambiguous text-label relation.** The complete review contains only these two apparently positive words, but the supplied label is negative. The text alone does not explain the rating.

Proposed validation check: Audit similarly short validation reviews against their source ratings where available; retain the official test label and report short-text uncertainty.

### 4. test_3163 - Confident false positive

Label negative; predicted positive; P(positive)=0.999997. Processed tokens: 6; retained at most 192.

> That is all I love though...

**Restricted praise.** The final sentence limits the praise to beer. The processed input is 'love rubios beer n nthat love', so the restriction is largely lost and praise is repeated.

Proposed validation check: Compare lighter stopword removal and sentence-aware pooling on validation reviews containing limited praise.

### 5. test_23566 - Confident false positive

Label negative; predicted positive; P(positive)=0.99999356. Processed tokens: 4; retained at most 192.

> fast,  a little toooo fast

**Implicit criticism.** The repeated speed word looks positive in isolation, but 'a little toooo fast' can criticize rushed service. The supplied negative label depends on that qualification.

Proposed validation check: Evaluate character or subword features and repeated-letter normalization on validation examples with informal intensifiers.

### 6. test_1131 - Confident false negative

Label positive; predicted negative; P(positive)=1.1857003e-08. Processed tokens: 4; retained at most 192.

> wont do ya dirty

**Informal negation and idiom.** The apostrophe-free 'wont' is not expanded by the contraction rule. The input 'margaritas wont ya dirty' also requires the positive idiom 'do no harm'.

Proposed validation check: Normalize common apostrophe-free contractions and test phrase or character features on informal validation text.

### 7. test_24208 - Confident false negative

Label positive; predicted negative; P(positive)=2.1770502e-06. Processed tokens: 3; retained at most 192.

> is the restaurant closed?

**Question rather than sentiment.** This asks about opening status rather than clearly expressing dissatisfaction. Processing leaves 'not restaurant closed', while the supplied label is positive.

Proposed validation check: Inspect question and business-status slices on validation data and distinguish factual closure mentions from sentiment.

### 8. test_22707 - Confident false negative

Label positive; predicted negative; P(positive)=2.4895903e-06. Processed tokens: 10; retained at most 192.

> but variety was just ok.... Meh..

**Mixed sentiment and rating ambiguity.** The review praises the view but ends with weak food praise. A negative prediction is understandable from the wording even though the supplied label is positive.

Proposed validation check: Audit mixed validation reviews and compare sentence-level aggregation without changing official test labels.

### 9. test_16998 - Confident false negative

Label positive; predicted negative; P(positive)=9.54978e-06. Processed tokens: 5; retained at most 192.

> are you gonna reopen?

**Implicit affection in a complaint.** Repeated requests to reopen can imply that the reviewer misses the business. The model instead receives words such as 'wtf' and 'reopen' with little explicit praise.

Proposed validation check: Evaluate a validation slice of reopening questions and informal expressions; compare phrase-aware features.

### 10. test_29123 - Confident false negative

Label positive; predicted negative; P(positive)=1.7520259e-05. Processed tokens: 4; retained at most 192.

> made me give it four stars

**Preprocessing: lost rating cue.** The explicit four-star explanation supports the positive label. Processing reduces the whole review to 'ok but bread stars', dropping the number word 'four'.

Proposed validation check: Preserve number words and explicit rating phrases, then test the change on validation data.

### 11. test_24137 - Near threshold

Label positive; predicted negative; P(positive)=0.4999231. Processed tokens: 78; retained at most 192.

> I recommend it

**Mixed sentiment and comparison.** Complaints about Vegas prices and portions surround a recommendation for this restaurant. Mean pooling cannot directly assign those statements to different comparison targets.

Proposed validation check: Compare sentence or attention pooling and evaluate mixed-review validation cases using the same training split.

### 12. test_12064 - Near threshold

Label negative; predicted positive; P(positive)=0.5004356. Processed tokens: 43; retained at most 192.

> It just wasn't worth the price

**Positive setting but negative value.** Praise for the atmosphere competes with tiny portions, high cost and a negative final verdict. The prediction is only just above the fixed threshold.

Proposed validation check: Test sentence-aware pooling that retains the final verdict; assess any threshold change on validation data only.

### 13. test_868 - Near threshold

Label negative; predicted positive; P(positive)=0.5006376. Processed tokens: 36; retained at most 192.

> We spent most of our night waiting in line

**Aspect-specific disappointment.** The drinks and company are praised, but the event organization is criticized. Averaging positive and negative words nearly cancels the overall complaint.

Proposed validation check: Compare sentence-level aggregation on validation reviews where the product is good but the experience is poor.

### 14. test_9057 - Near threshold

Label positive; predicted negative; P(positive)=0.49926805. Processed tokens: 92; retained at most 192.

> a near- Nirvana experience with every bite

**Positive outcome after inconvenience.** The long wait and brusque service are explicitly outweighed by the food. All 92 processed tokens fit, so this is not a truncation error.

Proposed validation check: Compare context-aware pooling on validation narratives whose conclusion reverses earlier complaints.

### 15. test_13888 - Near threshold

Label positive; predicted negative; P(positive)=0.49921444. Processed tokens: 130; retained at most 192.

> Overall, this is a good music venue

**Mixed aspects with positive conclusion.** The review recommends the venue despite seating complaints. All 130 processed tokens fit; missing tail text cannot explain this case.

Proposed validation check: Test sentence-level weighting while keeping the same split and inspect venue reviews with minor complaints.

### 16. test_9438 - Long-review / negation slice

Label positive; predicted negative; P(positive)=3.178318e-05. Processed tokens: 14; retained at most 192.

> never have to go to ghetto Fiesta Mall

**Wrong comparison target.** The negative language describes alternative malls, while the reviewer wants to keep using the reviewed mall. A bag of words loses this target distinction.

Proposed validation check: Evaluate comparison-target examples on validation data and compare a sequence-aware baseline.

### 17. test_6520 - Long-review / negation slice

Label positive; predicted negative; P(positive)=5.0243525e-05. Processed tokens: 17; retained at most 192.

> The beer is Delicious and so is the food

**Mixed aspects and rating ambiguity.** Positive food and atmosphere statements are followed by strong service complaints. The positive supplied label weights those aspects differently from the negative prediction.

Proposed validation check: Compare aspect or sentence aggregation and audit mixed validation labels instead of relabeling the test case.

### 18. test_22807 - Long-review / negation slice

Label positive; predicted negative; P(positive)=8.298872e-05. Processed tokens: 52; retained at most 192.

> They really did change the service up

**Updated review versus historical complaint.** A short positive edit precedes a much longer older complaint. All 52 processed tokens fit. Each model's selected error shows difficulty separating the update from the history.

Proposed validation check: Preserve update and sentence boundaries and compare update-aware pooling on validation examples; do not assume a longer sequence alone will fix it.

### 19. test_31102 - Long-review / negation slice

Label positive; predicted negative; P(positive)=9.7369295e-05. Processed tokens: 5; retained at most 192.

> has not closed

**Negation scope.** The review corrects a closure rumor and says the restaurant is open. 'not' survives preprocessing, so retaining the token alone did not make the mean-pooling model learn its scope.

Proposed validation check: Test negation-aware bigrams or a sequence model on a fixed validation slice of business-status corrections.

### 20. test_26639 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.00012118082. Processed tokens: 14; retained at most 192.

> Need a refund on our $20 voucher

**Ambiguous text-label relation.** The text describes closure and inconvenience, despite its positive supplied label. The reason for that label is not recoverable from this excerpt alone.

Proposed validation check: Audit similar validation labels against original ratings where available; retain this official test label and avoid claiming certain annotation error.


## TextCNN cases

Checkpoint: `task2_formal_20260927T020054Z_ba87cd57/textcnn/best.pt`, epoch 3.

[Annotated CSV](outputs/task2_formal_20260927T020054Z_ba87cd57/textcnn/error_review.csv) | [Selection and analysis status](outputs/task2_formal_20260927T020054Z_ba87cd57/textcnn/error_review_status.json)

### 1. test_37771 - Confident false positive

Label negative; predicted positive; P(positive)=0.99987245. Processed tokens: 31; retained at most 192.

> DO NOT EAT AT THIS LOCATION

**Chain praise versus location complaint.** Praise for the chain is followed by an explicit warning about this branch. All 31 processed tokens fit, so both selected models miss a target and contrast distinction rather than omitted text.

Proposed validation check: Test sentence-level target and contrast aggregation on validation reviews that compare a chain with one branch.

### 2. test_29330 - Confident false positive

Label negative; predicted positive; P(positive)=0.99978596. Processed tokens: 22; retained at most 192.

> Great place to come and relax worth a try!

**Ambiguous text-label relation.** The complete text is positive, but the supplied label is negative. There is no clear negative statement to use as an explanation.

Proposed validation check: Audit comparable validation examples against source ratings; leave official test labels unchanged and report the ambiguity.

### 3. test_11176 - Confident false positive

Label negative; predicted positive; P(positive)=0.99962795. Processed tokens: 8; retained at most 192.

> Ruined by music

**Desired state versus actual experience.** The poem says the place should be calm and tranquil, not that it was. Positive descriptive words refer to a desired state and may compete with the actual complaint.

Proposed validation check: Keep modal words and sentence boundaries; test this preprocessing change on hypothetical or desired-state validation reviews.

### 4. test_4655 - Confident false positive

Label negative; predicted positive; P(positive)=0.9994824. Processed tokens: 33; retained at most 192.

> totally the coolest place, like, ever

**Mixed sentiment and possible sarcasm.** The review lists good and bad points and mocks the crowd's excitement. Literal positive phrases could mislead max pooling; sarcasm is a plausible reading, not a measured attribution.

Proposed validation check: Compare sentence-aware aggregation on a reviewed validation subset containing irony and mixed pros and cons.

### 5. test_18122 - Confident false positive

Label negative; predicted positive; P(positive)=0.99938834. Processed tokens: 39; retained at most 192.

> sad to see them go

**Closure sadness versus quality praise.** The reviewer mourns a closure while repeatedly praising the food and service. The negative supplied label may reflect sadness rather than poor business quality.

Proposed validation check: Audit closure and farewell validation reviews and separate emotional tone from evaluation of the business.

### 6. test_16801 - Confident false negative

Label positive; predicted negative; P(positive)=0.0004381646. Processed tokens: 52; retained at most 192.

> will always return

**Mixed review with return intention.** The reviewer intends to return and likes the sandwich, but several sides and billing are criticized. The selected CNN prediction is dominated by an overall negative decision despite these positive commitments.

Proposed validation check: Compare sentence or attention pooling and evaluate return-intention validation examples without changing the test threshold.

### 7. test_30793 - Confident false negative

Label positive; predicted negative; P(positive)=0.0005658463. Processed tokens: 46; retained at most 192.

> Now its much better

**Past versus current experience.** The old owners receive strong criticism, whereas the new owners are praised at both ends of the review. All 46 processed tokens fit; this is a temporal contrast problem.

Proposed validation check: Keep temporal markers and compare sentence-aware context aggregation on before-and-after validation reviews.

### 8. test_16532 - Confident false negative

Label positive; predicted negative; P(positive)=0.00059241685. Processed tokens: 31; retained at most 192.

> Great gym, but

**Mixed sentiment and rating ambiguity.** A short positive opening is followed by a detailed cancellation and billing complaint. The supplied positive label is not an obvious summary of the full text.

Proposed validation check: Audit cancellation-related validation reviews and compare aspect-level summaries, keeping official labels fixed.

### 9. test_29494 - Confident false negative

Label positive; predicted negative; P(positive)=0.0008924034. Processed tokens: 165; retained at most 192.

> updating my review to 4 stars

**Update and service recovery.** The long billing complaint is followed by a successful resolution and an explicit positive rating update. All 165 processed tokens fit, so this cannot be attributed to the 192-token limit.

Proposed validation check: Preserve update and rating cues and compare sentence-level aggregation on validation cases with resolved complaints.

### 10. test_3223 - Confident false negative

Label positive; predicted negative; P(positive)=0.0009119775. Processed tokens: 20; retained at most 192.

> will probably go back again

**Mixed visits and return intention.** A good first visit and possible return coexist with a disappointing second visit. The wording gives genuine mixed evidence despite the positive label.

Proposed validation check: Evaluate multiple-visit validation reviews and compare sentence aggregation with max pooling.

### 11. test_33465 - Near threshold

Label positive; predicted negative; P(positive)=0.49974895. Processed tokens: 4; retained at most 192.

> Not bad all. Love the beers

**Negation scope.** The short review expresses praise through 'not bad'. The CNN output lies just below 0.5 even though the negation token is retained.

Proposed validation check: Compare negation-aware phrase features on validation text; select any calibration or threshold adjustment using validation labels only.

### 12. test_3618 - Near threshold

Label negative; predicted positive; P(positive)=0.50026375. Processed tokens: 34; retained at most 192.

> This company will rob local shoppers

**Promotion language inside a complaint.** The review quotes an attractive promotion to criticize its restrictions. Positive sales words describe the disputed offer, not the reviewer's approval.

Proposed validation check: Preserve quotation and sentence structure and compare context-aware pooling on promotion-complaint validation cases.

### 13. test_19442 - Near threshold

Label positive; predicted negative; P(positive)=0.49969393. Processed tokens: 252; retained at most 192.

> I'd try a few more tapases

**Mixed aspects with truncated ending.** The review mixes praise and criticism. Its 252 processed tokens exceed 192, excluding the conclusion and some service discussion; this is a plausible source of lost context.

Proposed validation check: Compare head-plus-tail input or a longer limit on validation data and report changes specifically on truncated reviews.

### 14. test_28747 - Near threshold

Label positive; predicted negative; P(positive)=0.49960852. Processed tokens: 81; retained at most 192.

> worst kept secret in Vegas

**Idiomatic wording and mixed aspects.** The opening uses a negative-looking idiom to describe a popular restaurant, then praises the pizza while noting queues and price. All 81 processed tokens fit.

Proposed validation check: Compare contextual phrase handling on validation examples with idioms, keeping the existing test set untouched.

### 15. test_13462 - Near threshold

Label negative; predicted positive; P(positive)=0.5005151. Processed tokens: 66; retained at most 192.

> If u want a guy threatening you

**Safety complaint among amenities.** Cheap drinks and friendly staff are mentioned, but threats are central to the negative conclusion. The prediction falls only slightly above 0.5.

Proposed validation check: Evaluate aspect and sentence weighting on validation reviews where a serious complaint outweighs minor amenities.

### 16. test_26172 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0014236559. Processed tokens: 80; retained at most 192.

> would eat here more often

**Wrong comparison target.** The harsh criticism concerns grocery-store instant ramen, while the restaurant is recommended. All 80 processed tokens fit, so target confusion is more plausible than truncation.

Proposed validation check: Test comparison-target validation cases with sentence or attention pooling.

### 17. test_20061 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0014349369. Processed tokens: 290; retained at most 192.

> Can't beat that!

**Domain vocabulary and truncated resolution.** Garbage and waste are literal descriptions of the service. Only 192 of 290 processed tokens are retained, omitting the final resolution and positive conclusion.

Proposed validation check: Compare head-plus-tail input on validation data and inspect non-restaurant domains separately before attributing all errors to length.

### 18. test_14906 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0015368646. Processed tokens: 69; retained at most 192.

> SouthPointe matched the lower price

**Past complaint versus successful resolution.** Much of the complaint concerns a prior booking site and last year's experience. The current hotel resolves the price problem; all 69 processed tokens fit.

Proposed validation check: Compare sentence-aware handling of entity and time changes on validation booking reviews.

### 19. test_22807 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0016308617. Processed tokens: 52; retained at most 192.

> They really did change the service up

**Updated review versus historical complaint.** A short positive edit precedes a much longer older complaint. All 52 processed tokens fit. Each model's selected error shows difficulty separating the update from the history.

Proposed validation check: Preserve update and sentence boundaries and compare update-aware pooling on validation examples; do not assume a longer sequence alone will fix it.

### 20. test_30958 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0017511233. Processed tokens: 193; retained at most 192.

> why do I give this store 5 stars?

**Company complaint versus local service recovery.** The local manager resolves a long complaint about the wider company. There are 193 processed tokens, but truncation removes only the final 'thank'; the positive resolution remains visible.

Proposed validation check: Test entity and resolution-aware sentence pooling on validation data. Do not treat a longer limit as a sufficient fix for this case.


## BiLSTM cases

Checkpoint: `task2_formal_20260927T020054Z_ba87cd57/bilstm/best.pt`, epoch 3.

[Annotated CSV](outputs/task2_formal_20260927T020054Z_ba87cd57/bilstm/error_review.csv) | [Selection and analysis status](outputs/task2_formal_20260927T020054Z_ba87cd57/bilstm/error_review_status.json)

### 1. test_10827 - Confident false positive

Label negative; predicted positive; P(positive)=0.99916995. Processed tokens: 88; retained at most 192.

> The music sucked

**Mixed aspects and emotional emphasis.** Praise for a dancer, lockers and one song competes with strong criticism of the club's music. All 88 processed tokens fit; the positive decision misses the review's weighting of those aspects.

Proposed validation check: Compare sentence or aspect aggregation on mixed club-review validation examples.

### 2. test_23764 - Confident false positive

Label negative; predicted positive; P(positive)=0.99894756. Processed tokens: 189; retained at most 192.

> it's not bad, but it's not amazing either

**Pros and cons with weak food verdict.** The review lists many amenities but criticizes food quality. All 189 processed tokens fit, so the long-review slice does not imply truncation in this example.

Proposed validation check: Preserve pros-and-cons boundaries and compare sentence-level aggregation on validation reviews with mixed food and service judgments.

### 3. test_2603 - Confident false positive

Label negative; predicted positive; P(positive)=0.99894446. Processed tokens: 45; retained at most 192.

> the prices are crazy high

**Positive exception within a negative comparison.** Shoes are praised as an exception, but prices and overall selection are criticized and competitors are preferred. The positive prediction misses that overall comparison.

Proposed validation check: Evaluate exception and competitor comparisons on validation data with sentence-aware pooling.

### 4. test_34683 - Confident false positive

Label negative; predicted positive; P(positive)=0.99884593. Processed tokens: 59; retained at most 192.

> They only have one issue, bland, plain food

**Many amenities versus a decisive food complaint.** Many positive amenities precede the central food complaint and a preference for other restaurants. All 59 processed tokens fit.

Proposed validation check: Compare aspect-aware aggregation on validation cases where one important weakness determines the overall rating.

### 5. test_18122 - Confident false positive

Label negative; predicted positive; P(positive)=0.99839395. Processed tokens: 39; retained at most 192.

> sad to see them go

**Closure sadness versus quality praise.** The reviewer mourns a closure while repeatedly praising the food and service. The negative supplied label may reflect sadness rather than poor business quality.

Proposed validation check: Audit closure and farewell validation reviews and separate emotional tone from evaluation of the business.

### 6. test_22807 - Confident false negative

Label positive; predicted negative; P(positive)=0.00015927476. Processed tokens: 52; retained at most 192.

> They really did change the service up

**Updated review versus historical complaint.** A short positive edit precedes a much longer older complaint. All 52 processed tokens fit. Each model's selected error shows difficulty separating the update from the history.

Proposed validation check: Preserve update and sentence boundaries and compare update-aware pooling on validation examples; do not assume a longer sequence alone will fix it.

### 7. test_10288 - Confident false negative

Label positive; predicted negative; P(positive)=0.000741355. Processed tokens: 68; retained at most 192.

> trying to make it right and apologizing

**Service recovery despite food complaints.** The reviewer rewards the staff's response despite disliking the food and water service. All 68 processed tokens fit; the model must weigh resolution against complaints.

Proposed validation check: Evaluate service-recovery validation reviews and compare sentence-level attention.

### 8. test_30958 - Confident false negative

Label positive; predicted negative; P(positive)=0.00078077434. Processed tokens: 193; retained at most 192.

> why do I give this store 5 stars?

**Company complaint versus local service recovery.** The local manager resolves a long complaint about the wider company. There are 193 processed tokens, but truncation removes only the final 'thank'; the positive resolution remains visible.

Proposed validation check: Test entity and resolution-aware sentence pooling on validation data. Do not treat a longer limit as a sufficient fix for this case.

### 9. test_30793 - Confident false negative

Label positive; predicted negative; P(positive)=0.00078439724. Processed tokens: 46; retained at most 192.

> Now its much better

**Past versus current experience.** The old owners receive strong criticism, whereas the new owners are praised at both ends of the review. All 46 processed tokens fit; this is a temporal contrast problem.

Proposed validation check: Keep temporal markers and compare sentence-aware context aggregation on before-and-after validation reviews.

### 10. test_9871 - Confident false negative

Label positive; predicted negative; P(positive)=0.0011509965. Processed tokens: 136; retained at most 192.

> TI is always a good value

**Narrative trouble with positive outcome.** The hotel resolves the reviewer's booking mistake and receives a positive overall judgment despite several inconveniences. All 136 processed tokens fit.

Proposed validation check: Compare sentence-level summaries on validation narratives with a positive outcome after an initially negative incident.

### 11. test_16678 - Near threshold

Label negative; predicted positive; P(positive)=0.500563. Processed tokens: 51; retained at most 192.

> I couldn't wait to leave

**Minor positive aspect versus negative verdict.** The children's enjoyment of a fountain is a small positive within a tourist-trap complaint. The prediction is just above the threshold despite the final negative verdict.

Proposed validation check: Test sentence-aware pooling on validation cases where the final verdict differs from one praised aspect.

### 12. test_28480 - Near threshold

Label negative; predicted positive; P(positive)=0.5007261. Processed tokens: 25; retained at most 192.

> cheaper, tastier, larger burger

**Wrong comparison target.** These positive adjectives describe the recommended competitor, while the reviewed business is called overpriced. The model crosses the threshold in the wrong direction.

Proposed validation check: Compare target-aware sentence representations on competitor-comparison validation reviews.

### 13. test_34705 - Near threshold

Label positive; predicted negative; P(positive)=0.49916467. Processed tokens: 11; retained at most 192.

> They're the tits

**Positive slang.** The final slang expression is praise in this context. The review is short and its positive meaning depends on recognizing that idiom.

Proposed validation check: Evaluate slang and idiom validation examples; compare character or phrase features learned only from training data.

### 14. test_32562 - Near threshold

Label positive; predicted negative; P(positive)=0.49902654. Processed tokens: 23; retained at most 192.

> Sandwiches are good

**Mixed service complaint and compensation.** Good sandwiches and free replacements coexist with repeated order mistakes and limited lunch time. The model is uncertain and lands just below 0.5.

Proposed validation check: Audit mixed service-recovery validation examples and evaluate sentence aggregation; do not tune the threshold on this test case.

### 15. test_31449 - Near threshold

Label positive; predicted negative; P(positive)=0.49883115. Processed tokens: 285; retained at most 192.

> deciding between 4 and 5 stars

**Mixed rated aspects and truncation.** The review gives an overall positive rating but mixed item ratings. Only 192 of 285 processed tokens are retained; omitted text contains both further criticism and beer praise, so truncation is not a proven sole cause.

Proposed validation check: Compare head-plus-tail input and retention of number-word rating cues on validation data, varying one choice at a time.

### 16. test_22206 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0012314891. Processed tokens: 115; retained at most 192.

> Serendipity, I will be back!

**Mixed dishes and explanatory negation.** An average quesadilla, a praised drink and a clarification of what the drink is not appear together. All 115 processed tokens fit, including the intention to return.

Proposed validation check: Compare sentence-aware aggregation on validation examples with product explanations and explicit return intent.

### 17. test_20061 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0012814221. Processed tokens: 290; retained at most 192.

> Can't beat that!

**Domain vocabulary and truncated resolution.** Garbage and waste are literal descriptions of the service. Only 192 of 290 processed tokens are retained, omitting the final resolution and positive conclusion.

Proposed validation check: Compare head-plus-tail input on validation data and inspect non-restaurant domains separately before attributing all errors to length.

### 18. test_37988 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0013292707. Processed tokens: 38; retained at most 192.

> However!!! I experienced something much better

**Expectation versus actual experience.** The opening negative terms describe expectations based on another branch; the actual visit is praised. All 38 processed tokens fit.

Proposed validation check: Preserve contrast markers and test expectation-versus-experience validation reviews with sentence-aware features.

### 19. test_30086 - Long-review / negation slice

Label positive; predicted negative; P(positive)=0.0015603743. Processed tokens: 53; retained at most 192.

> a good chef with a bad location

**Business closure versus product quality.** The review reports a closure and criticizes business planning while praising the chef. The supplied positive label and largely negative situation describe different aspects.

Proposed validation check: Audit closure-related validation examples and distinguish factual business status from food-quality sentiment.

### 20. test_37771 - Long-review / negation slice

Label negative; predicted positive; P(positive)=0.9981001. Processed tokens: 31; retained at most 192.

> DO NOT EAT AT THIS LOCATION

**Chain praise versus location complaint.** Praise for the chain is followed by an explicit warning about this branch. All 31 processed tokens fit, so both selected models miss a target and contrast distinction rather than omitted text.

Proposed validation check: Test sentence-level target and contrast aggregation on validation reviews that compare a chain with one branch.

## Next experiment, if more development is desired

First compare the current stopword list with a lighter rule that retains comparisons,
rating-number words and sentence boundaries. Separately test literal escape decoding
and common apostrophe-free contractions. A second experiment can compare the same
192-token budget split between the start and end with the present first-192 rule.
Use the same training/validation IDs, choose settings on validation, and keep the
reported test set and its labels unchanged. None of these extra experiments is
needed merely to finish documenting the completed run.
