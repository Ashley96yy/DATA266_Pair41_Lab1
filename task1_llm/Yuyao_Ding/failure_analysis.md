# Yuyao Ding - Task 1 Sequence Model Failure Analysis

Analyze three observed failures from the formal-run text samples in [notebook Section 1.3.5](src/task1_llm.ipynb). All samples use the best checkpoint (epoch 10) from run `train_20260927T000510Z_b9df2210` and contain 300 generated characters.

Source: [saved generation samples](outputs/train_20260927T000510Z_b9df2210/generations_20260927T002606Z_9466d7d0.json). Sample numbers below refer to the one-based order in that file. Abrupt endings caused by the generation limit are excluded from the failure analysis.

## 1.4.1 Case 1: Repetition

**Prompt:** `The dog wanted to`

**Decoding:** Greedy; sample 5.

**Generated snippet:**

> They were happy and sad. They had a big box of the box. They were happy and sad. They had a big box of the box. They were happy and sad.

**Observation:** The same sentence and box phrase repeat without introducing an event or advancing the story.

**Possible explanation:** Greedy decoding repeatedly selects the most likely next character. The resulting context returns to the same phrases here, which is consistent with a decoding loop.

## 1.4.2 Case 2: Broken Grammar

**Prompt:** `A little girl found`

**Decoding:** Temperature sampling (`temperature=0.8`, `seed=641`); sample 4.

**Generated snippet:**

> It's okay and you can buy it not mean to buy like to play with my friend.

**Observation:** The sequence "buy it not mean to buy like to play" joins several verb phrases without a clear grammatical relationship. Recognizable words do not form a well-structured sentence.

**Possible explanation:** The character model has learned common words and short phrases but does not reliably combine them into sentences. This example does not separate the effect of the trained model from the sampling method.

## 1.4.3 Case 3: Loss of Coherence

**Prompt:** `Once upon a time,`

**Decoding:** Temperature sampling (`temperature=0.8`, `seed=640`); sample 2.

**Generated snippet:**

> there was a little girl named Lily. The bird loved to take a special music. They were happy and loved to play together. One day, Lily's mom was delicious and called Ben.

**Observation:** A bird appears without being introduced, "they" has no clear referent, and Lily's mother is described as "delicious" without supporting context. The passage shifts subjects and meaning instead of developing one event.

**Possible explanation:** The model produces familiar story phrases without keeping their subjects and meanings consistent. Small capacity may contribute, but this sample alone does not identify the cause.

## 1.4.4 Comparison and Limitations

Across the three completions per method, the saved character-level repeated 4-gram rate is **63.08% for greedy decoding** and **26.71% for temperature sampling**. It is computed as `(total 4-grams - unique 4-grams) / total 4-grams`, excluding prompts and without creating 4-grams across sample boundaries. This supports the greater repetition observed in these greedy samples, but common character sequences can also contribute to this metric.

The sampled text still contains grammatical and coherence errors. These six short outputs illustrate specific failures; they do not establish a general ranking of decoding methods. The explanations above are hypotheses, not conclusions from controlled experiments.

Best-checkpoint SHA-256: `5cde3d36c7cb5b29224e19644e50be952a97c0e440c4cc3199169d6a2c8518ef`. Aggregate results and evaluation definitions are in [results.md](results.md); all required metrics are in [metrics_report.csv](metrics_report.csv).
