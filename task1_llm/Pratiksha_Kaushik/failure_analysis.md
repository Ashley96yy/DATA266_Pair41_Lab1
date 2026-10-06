# Task 1 — Failure Analysis (character-level GPT on TinyStories)

Member: pratiksha_kaushik  
Run: `20261005-223946_L8H8C512T512`  
Checkpoint: `checkpoints/ckpt_best.pt` (epoch 10, val CE 0.4742)

Each case was selected automatically by a measurable criterion (highest repeated 4-gram rate, highest invented-word rate, lowest first-half/second-half overlap), so the cases are not cherry-picked.

## Summary

| # | Failure type | Decoding setting | Key evidence |
|---|---|---|---|
| 1 | Repetition (degenerate looping) | greedy decoding (temperature = 0) | repeated 4-gram rate 86.6% vs 1.1% in real stories; distinct-2 of greedy set 0.379 vs 0.655 at t=0.8 |
| 2 | Broken grammar / invented words | temperature = 1.3 | invented words: ['aira', "gardeners'", 'originalt', 'svery']; invented-word rate 4.0% vs 0.03% at t=0.8 |
| 3 | Loss of coherence / entity drift | temperature = 0.8, 1533 chars (context window = 512) | content-word overlap between halves 0.074; names in first half ['Lily', 'Tom'], only in second half [] |

## Case 1: Repetition (degenerate looping)
**Setting:** greedy decoding (temperature = 0)  
**Evidence:** repeated 4-gram rate 86.6% vs 1.1% in real stories; distinct-2 of greedy set 0.379 vs 0.655 at t=0.8

```text
The sun was hot and the birds were singing. The birds were singing and the sun was shining. The birds were singing and the sun was shining.

The birds were singing and the sun was shining. The birds were singing and the sun was shining. The birds were singing and the sun was shining.

The birds were
```
**Observation:** Greedy decoding always takes the single most likely character, so once a high-probability phrase is produced the same context recurs and the model falls into a loop it cannot leave. The looped phrases are usually grammatical, high-frequency TinyStories phrases, so the failure is in the decoding strategy, not the per-step predictions.

## Case 2: Broken grammar / invented words
**Setting:** temperature = 1.3  
**Evidence:** invented words: ['aira', "gardeners'", 'originalt', 'svery']; invented-word rate 4.0% vs 0.03% at t=0.8

```text
The sun was hot and the birds were singing. Lucy wanted to get closer and she begged to ask them for help if they could. But her parents already said no to each other.

Peter was svery exhausted, but they decided to keep going. It seemed like someday it fit up heavily in her green home. When they opened one of their gardeners' doors, a strange noise filled t
```
**Observation:** At high temperature, probability mass moves into the tail, where low-likelihood characters are sampled in the middle of words. Because the model spells character by character, one bad sample produces a non-word, and each error makes the context less familiar, so errors compound. Sentence boundaries and agreement also break down (missing verbs, wrong pronouns).

## Case 3: Loss of coherence / entity drift
**Setting:** temperature = 0.8, 1533 chars (context window = 512)  
**Evidence:** content-word overlap between halves 0.074; names in first half ['Lily', 'Tom'], only in second half []

```text
Tom and his dog went to the park. They saw many things: trees, flowers, birds, dogs. They liked to play on the swings, the slide, and the sandbox. Sometimes they saw fish or crabs. They also saw fish swimming in the water.

"Look, Dad, look
[...]
ave to clean up this mess. Now we have to go to the sandbox. And then we can play with the sand together. You have to keep the sandbox. And you have to say sorry to the sandbox. I will keep the sand away."

She ran to the sandbox and waited. The sand was soft
```
**Observation:** Each sentence is locally fluent, but the story loses its thread. The model only sees the last 512 characters (about 102 words), so after a few sentences the opening (who the protagonist is, what problem was set up) has dropped out of the context. Characters get renamed or replaced and the ending does not resolve the beginning. This is a hard limit of the context window, not under-training.
