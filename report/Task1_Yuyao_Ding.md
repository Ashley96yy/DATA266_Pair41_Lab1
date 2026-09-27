# Task 1 - Yuyao Ding

This is Yuyao Ding's contribution for the combined report. The teammate comparison and jointly written analysis still need the teammate's results. The main run is `train_20260927T000510Z_b9df2210`; the full individual report is [results.md](../task1_llm/Yuyao_Ding/results.md).

## Model, data and training

The model is a character-level decoder with learned token and position embeddings, two manually implemented pre-normalized Transformer blocks, and an untied language-modelling head. Each block uses four causal attention heads, LayerNorm, a GELU feed-forward layer and residual connections. No prebuilt Transformer or attention module is used. A small model keeps this from-scratch experiment inexpensive; it was not chosen through an architecture search.

| Setting | Yuyao Ding |
| --- | --- |
| Parameters / vocabulary | 456,192 / 104 including `<UNK>` |
| Blocks / heads / hidden dimension / feed-forward dimension | 2 / 4 / 128 / 512 |
| Context / dropout / LayerNorm epsilon | 256 characters / 0.1 / 1e-5 |
| Dataset | Original TinyStories train and validation files |
| Selected stories | 100,000 training; 10,000 validation |
| Sampling seeds / model seed | 640 and 641 / 640 |
| Input-target windows | 256 inputs, targets shifted by one character; stride 256 |
| Batch size / epochs / optimizer steps | 64 / 10 / 54,200 |
| Optimizer | AdamW, betas=(0.9, 0.95), weight decay=0.01 |
| Learning rate | 2,710-step linear warm-up to 0.0003; cosine decay to 0.00003 |
| Gradient clipping / precision | Global L2 threshold 1.0 / float32 |
| CPU / GPU | Intel Core i9-14900KF / NVIDIA GeForce RTX 4090 |
| Python / NumPy / PyTorch | 3.12.14 / 2.5.3 / 2.8.0+cu128 |
| Best checkpoint | Epoch 10 |

The 100K/10K dataset sizes are interpreted as stories. Sampling preserves the official split and excludes selected training-story hashes from validation. The vocabulary is built on training text only. The CPU model was checked after the run; the original manifest retains its generic platform identifier. [Preprocessing details and hashes](../reproducibility/manifests/Yuyao_Ding/preprocessing_manifest.json), [run manifest](../reproducibility/manifests/Yuyao_Ding/train_20260927T000510Z_b9df2210/manifest.json) and [saved training source](../reproducibility/manifests/Yuyao_Ding/train_20260927T000510Z_b9df2210/source_snapshot.py) identify the experiment.

## Results

Both splits below use the same best checkpoint with dropout disabled. CE is token-weighted over each full selected split; perplexity is `exp(CE)` and bits-per-character is `CE / ln(2)`.

| Metric | Training | Validation |
| --- | ---: | ---: |
| Cross-entropy (nats/character) | 0.882596 | 0.882114 |
| Perplexity | 2.417166 | 2.416001 |
| Bits-per-character | 1.273317 | 1.272621 |
| Top-1 next-character accuracy | 72.2687% | 72.3116% |

Validation CE minus training CE is **-0.000482382 nats/character**. Validation loss fell each epoch and improved only slightly over the last three epochs (0.887282, 0.883779, 0.882114). The run approaches a plateau under this schedule, with no visible positive generalization gap. This is not a held-out test result.

![Training and validation loss](../task1_llm/Yuyao_Ding/outputs/train_20260927T000510Z_b9df2210/loss_curves.png)

The training curve averages changing weights with dropout active. It is not used to compute the matched-checkpoint gap above.

Three prompts were decoded using both greedy decoding and temperature 0.8, for 300 new characters each. Seeds were 640, 641 and 642. N-grams are pooled within each method, excluding prompts and never crossing sample boundaries.

| Generation metric | Greedy | Temperature 0.8 |
| --- | ---: | ---: |
| Distinct-1 | 0.038889 | 0.045556 |
| Distinct-2 | 0.189521 | 0.269788 |
| Distinct-3 | 0.300895 | 0.564877 |
| Repeated 4-gram rate | 0.630752 | 0.267116 |
| Generation characters/sec | 271.9974 | 247.3769 |

Distinct-n is unique/total character n-grams; repeated-4 is `(total - unique) / total`. Throughput divides 900 generated characters by total measured seconds for each method. There was no separate timing warm-up, so this is not a controlled decoder-speed comparison.

| Stability and cost | Value |
| --- | ---: |
| Gradient L2 norm: mean / median / max, before clipping | 0.715206 / 0.637479 / 4.003296 |
| Steps above the clipping threshold | 3,922 |
| Loss spikes / nonfinite loss or gradient steps | 0 / 0 |
| Training target characters/sec | 746664.1530 |
| Training-loop seconds | 1189.194 |
| Full training/evaluation wall seconds | 1247.819 |
| Process-lifetime peak RSS | 2,362,290,176 bytes |
| Peak PyTorch CUDA allocated memory | 750,842,880 bytes |

The post-hoc loss-spike rule is a minibatch CE above twice the median of the previous 100 minibatches within the same epoch; its first 100 steps are excluded. CUDA allocation is not reserved memory or total device memory. RSS covers the process lifetime. Training-loop time excludes evaluation and checkpoint saving; full wall time includes them but excludes preprocessing and later generation.

Full-precision values for all required metrics are in [metrics_report.csv](../task1_llm/Yuyao_Ding/metrics_report.csv), selected by the main `model_id`. Each row links its evidence. The [raw log](../reproducibility/raw_logs/Yuyao_Ding/train_20260927T000510Z_b9df2210.jsonl), [metrics](../task1_llm/Yuyao_Ding/outputs/train_20260927T000510Z_b9df2210/metrics.json), [generation samples](../task1_llm/Yuyao_Ding/outputs/train_20260927T000510Z_b9df2210/generations_20260927T002606Z_9466d7d0.json) and [report calculations](../task1_llm/Yuyao_Ding/outputs/train_20260927T000510Z_b9df2210/report_audit.json) are retained.

## Three failure cases

The snippets below are exact excerpts from the six recorded outputs, not newly generated examples. See [failure_analysis.md](../task1_llm/Yuyao_Ding/failure_analysis.md) for their observations and possible explanations.

1. **Repetition** - `The dog wanted to`, greedy, sample 5:

   > They were happy and sad. They had a big box of the box. They were happy and sad. They had a big box of the box. They were happy and sad.

   The passage repeats the same phrases without advancing the story.

2. **Broken grammar** - `A little girl found`, temperature 0.8, seed 641, sample 4:

   > It's okay and you can buy it not mean to buy like to play with my friend.

   Several verb phrases are joined without a clear grammatical relationship.

3. **Loss of coherence** - `Once upon a time,`, temperature 0.8, seed 640, sample 2:

   > there was a little girl named Lily. The bird loved to take a special music. They were happy and loved to play together. One day, Lily's mom was delicious and called Ben.

   The bird is not introduced, the pronoun is unclear, and the description of Lily's mother does not fit the story. Fixed-length truncation is not counted as a failure.

## Limitations and next steps

Temperature sampling reduced the measured repetition in these six outputs, but did not remove grammar and coherence errors. Character accuracy does not measure narrative quality. Model capacity and context length may constrain performance; neither was isolated in an ablation.

The previous Apple M3 run used the same saved configurations and preprocessing hashes. Its validation CE was 0.876024 and its wall time was 133.148 minutes, compared with 0.882114 and 20.797 minutes here. The observed speedup is 6.40 times, while validation quality is slightly lower. Different software environments, portability fixes and unavailable historical source/weights prevent treating this as a hardware-only experiment.

A useful next experiment would vary capacity or context length one at a time while keeping the data split and evaluation fixed. These are proposed experiments, not completed results. The teammate's actual model and metrics are still needed to finish the required team comparison and joint discussion.

## Artifact and demo

Best checkpoint: `task1_llm/Yuyao_Ding/checkpoints/train_20260927T000510Z_b9df2210/epoch_010.pt`.

SHA-256: `5cde3d36c7cb5b29224e19644e50be952a97c0e440c4cc3199169d6a2c8518ef`.

The [Task 1 guide](../task1_llm/Yuyao_Ding/README.md) explains environment setup, the one-command smoke test, checkpoint restoration and direct text-generation demo. The saved checkpoint and reproduction materials are included in the combined resource ZIP.

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Eldan and Li (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
