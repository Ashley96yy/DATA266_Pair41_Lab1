# Yuyao Ding - Task 1 Sequence Model Failure Analysis

Analyze three observed failures from the formal-run text samples in [notebook Section 1.3.5](src/task1_llm.ipynb). All samples use the best checkpoint (epoch 10) from run `train_20260914T211231Z_98b2aa86` and contain 300 generated characters.

Source: [saved generation samples](outputs/train_20260914T211231Z_98b2aa86/generations_20260914T232627Z_059e7b30.json). Sample numbers below refer to the one-based order in that file. Abrupt endings caused by the generation limit are excluded from the failure analysis.

## 1.4.1 Case 1: Repetition

**Prompt:** `A little girl found`

**Decoding:** Greedy; sample 3.

**Generated snippet:**

> I will be careful. I will be careful. I will be careful. I will be careful.

**Observation:** The same sentence appears four times consecutively in the box's dialogue. It adds no new information and stops the story from progressing.

**Possible explanation:** Greedy decoding selects the highest-probability next character at every step. In this sample, the resulting context repeatedly leads back to the same phrase. This is consistent with a repetitive decoding loop; it does not establish that greedy decoding always causes repetition.

## 1.4.2 Case 2: Broken Grammar

**Prompt:** `Once upon a time,`

**Decoding:** Temperature sampling (`temperature=0.8`, `seed=640`); sample 2.

**Generated snippet:**

> but it was a for to collect from her hard to make them a map story.

**Observation:** The phrase "a for to collect" has no grammatical structure, and "from her hard" does not form a clear relationship between the words. Although the individual words are recognizable, the sentence has no clear meaning.

**Possible explanation:** The character-level model can produce familiar words without consistently organizing them into grammatical sentences. Sampling allows alternative next characters, but this single example cannot establish whether sampling or the learned model is the main cause of the error.

## 1.4.3 Case 3: Loss of Coherence

**Prompt:** `The dog wanted to`

**Decoding:** Temperature sampling (`temperature=0.8`, `seed=642`); sample 6.

**Generated snippet:**

> Lily was happy to have a good of fun together.

**Observation:** The sample begins with a dog, then uses "they" without a clear referent, and later introduces Lily without explaining her connection to the dog or preceding events. The quoted sentence illustrates this unexplained character transition. The failure is the missing narrative connection, not merely the appearance of a new character.

**Possible explanation:** The model does not maintain a consistent narrative focus in this sample. Generation uses only the most recent 256 characters, which may limit access to earlier context as text grows. However, this example alone does not prove that context cropping caused the transition.

## 1.4.4 Comparison and Limitations

Across the three completions per method, the saved character-level repeated 4-gram rate is **62.40% for greedy decoding** and **30.08% for temperature sampling**. It is computed as `(total 4-grams - unique 4-grams) / total 4-grams`, excluding prompts and without creating 4-grams across sample boundaries. This supports the greater repetition observed in these greedy samples, but common character sequences can also contribute to this metric.

The sampled text still contains grammatical and coherence errors. These six short outputs illustrate specific failures; they do not establish a general ranking of decoding methods. The explanations above are hypotheses, not conclusions from controlled experiments.

Best-checkpoint SHA-256: `23933799df14748236f4a89d726f347a41485ba1885a9a654916d5be51c2eac6`. Aggregate results and evaluation definitions are in [results.md](results.md); all required metrics are in [metrics_report.csv](metrics_report.csv).
