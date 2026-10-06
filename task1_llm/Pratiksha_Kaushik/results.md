# Task 1 — Character-Level GPT on TinyStories: Results

**Member:** pratiksha_kaushik  
**Run:** `20261005-223946_L8H8C512T512`  
**Reported checkpoint:** `checkpoints/ckpt_best.pt` (epoch 10, step 27,360)  
**Raw log:** `reproducibility/raw_logs/task1_pratiksha_kaushik_20261005-223946_L8H8C512T512_runlog.txt`  
**Manifest:** `reproducibility/manifests/task1_pratiksha_kaushik_20261005-223946_L8H8C512T512_manifest.json`

## 1. Model Architecture

A decoder-only, GPT-style Transformer written from tensor operations. No `nn.Transformer`, `nn.MultiheadAttention` or
`scaled_dot_product_attention` is used. Attention, the causal mask, LayerNorm, the feed-forward network and the residual wiring
are all implemented in the notebook (`src/`), and two tests check the causal mask: the attention weights above the
diagonal are exactly zero, and editing future tokens leaves earlier logits unchanged.

| component | configuration | parameters |
|---|---|---|
| Token embedding | learnable, 80 × 512 | 40,960 |
| Positional embedding | learnable absolute, 512 × 512 | 262,144 |
| Transformer blocks | 8 × pre-LN [LayerNorm → causal MHSA → residual, LayerNorm → FFN → residual] | 25,219,072 (3,152,384 each) |
| Multi-head self-attention | 8 heads × 64 dims, fused QKV projection, lower-triangular causal mask, fp32 softmax | in blocks |
| Feed-forward network | 512 → 2048 → 512, GELU | in blocks |
| Final LayerNorm | 512 | 1,024 |
| Language-modelling head | linear 512 → 80, no bias, untied | 40,960 |
| Total |  | 25,564,160 |

## 2. Hyperparameters

| hyperparameter | value |
|---|---|
| Context length (block_size) | 512 |
| Layers | 8 |
| Heads | 8 |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Vocabulary | 80 (chars with freq ≥ 20 + `<|eos|>`, `<unk>`) |
| Batch size | 64 sequences = 32,768 tokens/step |
| Epochs | 10 |
| Steps | 27,360 (2,736/epoch) |
| Tokens seen | 896,532,480 |
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 on matrices only |
| Learning rate | peak 0.0008, linear warm-up 820 steps (3%), cosine decay to 8e-05 |
| Gradient clipping | global L2 norm 1.0 |
| Precision | torch.bfloat16 autocast + torch.compile |
| Seed | 1337 |
| Data split | 100,000 train / 10,000 val stories from the TinyStories train partition |

## 3. Training Decisions and Justification

| decision | justification |
|---|---|
| Character-level tokenisation | Required by the task. With 80 symbols the embedding and output layers are tiny (344,064 params), so almost all capacity goes into the Transformer blocks. |
| Personal split with deduplication | Stories are drawn by a seeded permutation of the train partition, normalised and deduplicated before splitting, so no story appears in both train and validation. The indices are saved in `data_processed/split_indices.json`. |
| Text normalisation (ftfy + ASCII punctuation) | TinyStories contains mojibake (`â€™`) and several kinds of quote and dash. Repairing and mapping them keeps the vocabulary small and stops the model wasting capacity on encoding noise. Paragraph breaks are kept. |
| `<|eos|>` story separator | Lets the model learn where stories begin and end, and lets generation start a fresh story from `<|eos|>` alone. |
| Non-overlapping windows with a random offset per epoch | Every training character is seen exactly once per epoch, while window boundaries change between epochs, which acts as cheap augmentation. |
| Pre-LayerNorm blocks + scaled residual init | Keeps the residual stream an identity path, so an 8-layer model trains stably without a long warm-up. Output projections use std 0.02/√(2·8) so activations don't grow with depth (GPT-2). |
| Model size 8L/512d, context 512 | A pilot 6L/384d model with a 256-character context (on an earlier preprocessing version) ended at val CE 0.544 with a train/val gap of only 0.016 and the loss still falling, so it was capacity-limited rather than overfitting. The larger width and depth add capacity, and the longer context covers more of each story. |
| AdamW, β₂ = 0.95, decay on matrices only | β₂ = 0.95 reacts faster to gradient-scale changes than 0.999 and is standard for GPT training. Biases and LayerNorm gains are not decayed because shrinking them towards zero has no regularising benefit. |
| Linear warm-up + cosine decay | Warm-up (820 steps) avoids large early updates while Adam's moment estimates are unreliable. Cosine decay to 8e-05 anneals into a lower-loss region at the end of training. |
| Gradient clipping at 1.0 | A safety net against rare large updates; 1.06% of steps were clipped, almost all of them during warm-up. |
| Dropout 0.1 | Light regularisation, since the model sees each training character 10 times. |
| torch.bfloat16 mixed precision + torch.compile | Higher throughput and lower memory use with no change to the maths. The softmax and the loss are computed in fp32. |
| Checkpoint selection by full validation CE | `ckpt_best.pt` is chosen on the complete validation set, not a subset, and every reported number comes from it. |

## 4. Results and Metrics

All required metrics, with units and settings, are in `metrics_report.csv`.

| metric | value | unit | setting |
|---|---|---|---|
| Training cross-entropy loss | 0.4476 | nats/char | train split, dropout off |
| Validation cross-entropy loss | 0.4742 | nats/char | full val split |
| Perplexity (validation) | 1.6067 |  | exp(val CE) |
| Perplexity (training) | 1.5645 |  | exp(train CE) |
| Bits-per-character (validation) | 0.6841 | bits/char | val CE / ln 2 |
| Bits-per-character (training) | 0.6457 | bits/char | train CE / ln 2 |
| Generalization gap | 0.0266 | nats/char | val CE - train CE |
| Generalization gap (relative) | 5.9510 | % | val CE / train CE - 1 |
| Top-1 next-character accuracy (validation) | 84.7353 | % | full val split |
| Top-1 next-character accuracy (training) | 85.4367 | % | train split |
| Top-1 accuracy, word start targets (validation) | 56.3250 | % | 19.6% of targets |
| Top-1 accuracy, inside word targets (validation) | 92.0341 | % | 56.4% of targets |
| Top-1 accuracy, space / punctuation targets (validation) | 90.5243 | % | 24.0% of targets |
| Gradient norm mean | 0.1541 | L2 | post warm-up, pre-clip |
| Gradient norm median | 0.1449 | L2 | post warm-up, pre-clip |
| Gradient norm p99 | 0.3409 | L2 | post warm-up, pre-clip |
| Gradient norm max | 25.5321 | L2 | all steps, pre-clip |
| Steps clipped | 1.0563 | % | grad norm > 1.0 |
| Loss spikes | 0 | count | loss > 1.3 x EMA after warm-up |
| NaN / Inf steps | 0 | count | non-finite loss or gradient |
| Parameter count (total) | 25,564,160 | params |  |
| Parameter count (non-embedding) | 25,261,056 | params |  |
| Training throughput | 452,013 | tokens/s | steady-state optimizer steps |
| Generation throughput (single stream) | 96 | tokens/s | batch 1, no KV-cache |
| Generation throughput (batch 32) | 2,182 | tokens/s | batch 32, no KV-cache |
| Peak GPU memory (allocated) | 10.5868 | GiB | torch.cuda.max_memory_allocated |
| Peak GPU memory (reserved) | 10.7891 | GiB | torch.cuda.max_memory_reserved |
| Total training time | 36.9107 | min | 10 epochs incl. evaluation |
| Epochs trained | 10 | epochs | 27,360 steps |
| Best epoch (val CE) | 10 | epoch |  |

### Per-epoch results

| epoch | train CE (running) | train CE | val CE | gap | val PPL | val BPC | val top-1 |
|---|---|---|---|---|---|---|---|
| 1 | 1.0368 | 0.6337 | 0.6369 | +0.0032 | 1.891 | 0.9189 | 79.72% |
| 2 | 0.6248 | 0.5690 | 0.5766 | +0.0076 | 1.780 | 0.8319 | 81.55% |
| 3 | 0.5796 | 0.5363 | 0.5480 | +0.0116 | 1.730 | 0.7906 | 82.46% |
| 4 | 0.5537 | 0.5142 | 0.5292 | +0.0151 | 1.698 | 0.7635 | 83.04% |
| 5 | 0.5341 | 0.4969 | 0.5147 | +0.0179 | 1.673 | 0.7426 | 83.49% |
| 6 | 0.5195 | 0.4832 | 0.5016 | +0.0184 | 1.651 | 0.7236 | 83.89% |
| 7 | 0.5047 | 0.4706 | 0.4913 | +0.0206 | 1.634 | 0.7087 | 84.20% |
| 8 | 0.4910 | 0.4596 | 0.4825 | +0.0229 | 1.620 | 0.6961 | 84.46% |
| 9 | 0.4806 | 0.4514 | 0.4766 | +0.0252 | 1.611 | 0.6876 | 84.65% |
| 10 | 0.4732 | 0.4470 | 0.4742 | +0.0272 | 1.607 | 0.6841 | 84.74% |

### Generation diversity (equal budget of 2,257 words per set)

| source | distinct-1 | distinct-2 | distinct-3 | rep. 4-gram | invented words | mean chars |
|---|---|---|---|---|---|---|
| model (t=0.8) | 0.2118 | 0.6546 | 0.8688 | 1.35% | 0.03% | 715 |
| model (greedy) | 0.1254 | 0.3788 | 0.5043 | 5.92% | 0.00% | 697 |
| real val stories | 0.2428 | 0.7230 | 0.9183 | 1.09% | 0.08% | 864 |

### Accuracy by target type (validation subset)

| target type | share of targets | top-1 accuracy |
|---|---|---|
| word start | 19.6% | 56.33% |
| inside word | 56.4% | 92.03% |
| space / punctuation | 24.0% | 90.52% |

### Loss curves and training stability

![loss curves](outputs/plots/loss_curves.png)

![stability](outputs/plots/stability.png)

### Example generation (τ = 0.8, prompt "Once upon a time")

```text
Once upon a time, there was a soft dog named Spot. Spot loved to run and play with his friends. One day, they were playing outside when they saw a big, mean truck driving by. The truck was loading water on the dog's fence. Spot was very scared and also wanted to run away.

Spot ran inside the truck and started to chase his friends. They ran and ran, but the truck was too fast. Spot was getting tired and did not want to stop. So, he sat on the fence and closed his eyes.

A kind man saw Spot and came outside. "I
```

More samples are in `outputs/samples/`, and the human-audit sheet is `outputs/human_audit.csv`.

## 5. Hardware Used

| item | value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU memory | 39 GiB |
| CUDA / cuDNN | 13.0 / 92700 |
| PyTorch / Python | 2.11.0+cu130 / 3.13.15 |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| Peak memory (allocated / reserved) | 10.59 / 10.79 GiB |
| Training time | 36.9 min (10 epochs incl. evaluation) |
| Training throughput | 452,013 tokens/s |
| Generation throughput | 96 tokens/s single stream, 2,182 tokens/s batch 32 |

## 6. Interpretation of Results

**Convergence.** Validation CE fell from 0.6369 after epoch 1 to 0.4742 after epoch 10 (25.5% lower), and the final epoch still improved it by 0.0024. The curve has flattened as the learning rate annealed, so more epochs at this size would give only small gains.

**Generalisation.** The final gap is +0.0266 nats/char (+5.95%), changing by +0.0240 between epoch 1 and the last epoch. This is small, so the model is not memorising the training stories. Validation loss improved every epoch, so the best checkpoint is the last one.

**What the numbers mean.** 0.684 bits per character means the model is on average about as uncertain as a choice between 1.61 equally likely characters. Its top-1 guess is right 84.7% of the time. The errors are concentrated at word starts (56.3% accuracy) rather than inside words (92.0%): once a word has started the model spells it almost deterministically, and most of the remaining uncertainty is in choosing the next word. Much of that uncertainty is inherent to story writing, not model error.

**Generation quality.** With τ = 0.8 sampling, distinct-1/2/3 are 0.212/0.655/0.869 against 0.243/0.723/0.918 for real stories at the same word budget, and the repeated 4-gram rate is 1.35% (real stories 1.09%). So sampled text is close to the diversity of the data, while greedy decoding repeats far more (5.9% repeated 4-grams). The invented-word rate at τ = 0.8 is 0.03%, which shows the model has learned the spelling of the TinyStories vocabulary.

**Stability and efficiency.** Loss spikes: 0; non-finite steps: 0. The median pre-clip gradient norm after warm-up was 0.145. Training ran at 452,013 tokens/s with a peak of 10.6 GiB. Generation is slower per token (96 tokens/s for a single stream) because each step re-encodes the full context window without a KV-cache. Batching amortises this (2,182 tokens/s at batch 32).

**Limits.** Local fluency is good, but long stories drift because the context window covers only part of a story. See `failure_analysis.md` for the three analysed failure cases (repetition, invented words, loss of coherence).
