# Checkpoints - Task 2 (Pratiksha Kaushik)

The model files are not in git. Download them from Google Drive and put them in this folder:

**Download:** https://drive.google.com/drive/folders/1sBI2vdiaSnv6S1KcWrIeYHQg5oVAT_k4?usp=drive_link

| File | Model | Threshold | Test accuracy | Parameters |
|---|---|---|---|---|
| `baseline_mean_embedding.pt` | mean-embedding baseline | 0.53 | 0.8914 | 3,840,129 |
| `experimental_cnn.pt` | CNN, kernels 3/5/7 (best model) | 0.56 | 0.9054 | 5,107,969 |
| `experimental_bigru.pt` | bidirectional GRU | 0.27 | 0.8880 | 5,022,977 |

Each file is a dict with `model_name`, `state_dict`, `threshold` and `seed` (9002), saved after epoch 5 of the final run in `src/task2_yelp_sentiment_all_models.ipynb`.

To load one, run the notebook up to the model definitions (section 2.2.4) so the vocabulary and model classes exist, then:

```python
ckpt = torch.load("checkpoints/experimental_cnn.pt", map_location=DEVICE)
model = model_factories[ckpt["model_name"]]().to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
threshold = ckpt["threshold"]
```
