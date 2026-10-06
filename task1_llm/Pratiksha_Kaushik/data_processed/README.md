# Processed data - Task 1 (Pratiksha Kaushik)

This is my own TinyStories split and character encoding, made by `src/Task1_gpt.ipynb` (run `20261005-223946_L8H8C512T512`, seed 1337).

**Download (all Task 1 files):** https://drive.google.com/drive/folders/19nN7gGNHc0h4IRYNC6TQ93BOfXxahAct?usp=drive_link

The `.npy` files are too large for the repo. The three small JSON files are in git.

| File | What it is | Size | SHA-256 |
|---|---|---|---|
| `vocab.json` | `idx_to_char` for all 80 symbols, plus `eos_id` (0) and `unk_id` (1). `char_to_idx` is the reverse of this map. | 1.1 KB | `a733de84e7f1f6302fd9e969ef6404994281ce2e2f5579123fefc2ae681d4d74` |
| `split_indices.json` | Row numbers in the TinyStories train partition used for my split: 100,000 train and 10,000 validation stories (seed 1337) | 0.9 MB | `597c9d3ee61e26eea03225ef93554a36c4fd66d30432baf9adae1d586719c269` |
| `dataset_stats.json` | Character counts, unknown-character rate, vocab size, dtype | 0.2 KB | `123a2264b1065359e2a7b98054eafa53ef1079c4554ef881d4e3d3482dc55e88` |
| `train_ids.npy` | All 100,000 training stories encoded as one uint8 array of character ids (89,656,644 characters) | 90 MB | `b0d15b29beedf94d9120f57f9140da01772c90f645d759a24734afa6272d7f1f` |
| `val_ids.npy` | All 10,000 validation stories encoded the same way (8,991,471 characters) | 9 MB | `47cafb636baeccc44a269c6f16890dfc22a65de29e803eb46ee2d675ee18049e` |

## How the data was made

1. Load `roneneldan/TinyStories` (2,119,719 stories in the train partition).
2. Shuffle with seed 1337. Fix broken text encoding with ftfy, map curly quotes and dashes to plain ASCII, drop stories shorter than 50 characters, and remove duplicates. Take the first 100,000 for training and the next 10,000 for validation. No story is in both.
3. Build the vocabulary from the training split: every character seen at least 20 times (78 characters), plus `<|eos|>` and `<unk>`. That gives 80 symbols, which cover 99.9999% of training characters. Unknown-character rate: 0.00013% train, 0.00021% validation.
4. Join the stories with `<|eos|>` between them and encode them as uint8 ids.
5. During training, the encoded text is cut into fixed-length input/target windows of 512 characters, with the target shifted one character to the right. That gives 175,109 windows per epoch, and the starting offset changes randomly each epoch.

Load in Python:

```python
import json, numpy as np
vocab = json.load(open("data_processed/vocab.json"))
idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
char_to_idx = {c: i for i, c in idx_to_char.items()}
train_ids = np.load("data_processed/train_ids.npy")
```
