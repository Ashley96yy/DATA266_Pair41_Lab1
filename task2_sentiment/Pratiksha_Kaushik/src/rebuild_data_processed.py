"""Rebuilds data_processed/ (split_indices.json, vocab.json, dataset_stats.json).

Re-runs the preprocessing cells of task2_yelp_sentiment_all_models.ipynb (sections 2.1.2-2.1.5) unchanged,
then checks the test split against outputs/predictions_experimental_cnn.csv.

Usage (from task2_sentiment/Pratiksha_Kaushik/):  python src/rebuild_data_processed.py
Needs: pandas, scikit-learn, pyarrow, huggingface_hub.
"""
import re, json, glob
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

import sys, os
from huggingface_hub import hf_hub_download
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_processed')
os.makedirs(OUT, exist_ok=True)
PRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'predictions_experimental_cnn.csv')
files = {s: hf_hub_download('fancyzhx/yelp_polarity', f'plain_text/{s}-00000-of-00001.parquet', repo_type='dataset') for s in ('train', 'test')}
train_raw = pd.read_parquet(files['train'])
test_raw = pd.read_parquet(files['test'])

SEED = 9002; MAX_TRAIN = 20_000; MAX_TEST = 5_000; VAL_SIZE = 0.10; MAX_VOCAB = 30_000; USE_STEMMING = False
STOPWORDS = {"a","an","and","are","as","at","be","but","by","for","from","has","have","he","her",
    "i","if","in","is","it","its","me","my","of","on","or","our","she","so","that","the",
    "their","there","they","this","to","was","we","were","what","when","which","who",
    "will","with","you","your"}
def light_stem(token):
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token
def tokenize(text):
    text = re.sub(r"[^a-z0-9\s' ]", " ", str(text).lower())
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    words = [w for w in words if w not in STOPWORDS]
    return [light_stem(w) for w in words] if USE_STEMMING else words
def clean_frame(frame):
    out = frame[["text", "label", "parquet_row"]].copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    out = out[out["label"].isin([0, 1])]
    out["tokens"] = out["text"].map(tokenize)
    return out[out["tokens"].map(len) > 0].reset_index(drop=True)

train_raw["parquet_row"] = np.arange(len(train_raw)); test_raw["parquet_row"] = np.arange(len(test_raw))
train_df = clean_frame(train_raw); test_df = clean_frame(test_raw)
print('after cleaning:', len(train_df), len(test_df), '(dropped', 560000-len(train_df), 38000-len(test_df), ')')
train_df, _ = train_test_split(train_df, train_size=MAX_TRAIN, stratify=train_df["label"], random_state=SEED)
test_df, _ = train_test_split(test_df, train_size=MAX_TEST, stratify=test_df["label"], random_state=SEED)
train_df, val_df = train_test_split(train_df.reset_index(drop=True), test_size=VAL_SIZE, stratify=train_df["label"], random_state=SEED)
train_df = train_df.reset_index(drop=True); val_df = val_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
print('splits:', len(train_df), len(val_df), len(test_df))

PAD, UNK = "<pad>", "<unk>"
counts = Counter(t for row in train_df["tokens"] for t in row)
vocab_tokens = [PAD, UNK] + [t for t, _ in counts.most_common(MAX_VOCAB - 2)]
print('vocab', len(vocab_tokens), '| most common:', counts.most_common(5))

# check against the saved test predictions (row order + labels) of the final run
pred = pd.read_csv(PRED)
same = (pred['label'].to_numpy() == test_df['label'].astype(int).to_numpy()).all()
print('test labels match saved predictions row by row:', same)
json.dump({'train': train_df['parquet_row'].tolist(), 'validation': val_df['parquet_row'].tolist(), 'test': test_df['parquet_row'].tolist(),
           'note': 'row numbers in the original fancyzhx/yelp_polarity train/test parquet files (train rows for train and validation, test rows for test)'},
          open(OUT + '/split_indices.json', 'w'))
json.dump({'stoi': {t: i for i, t in enumerate(vocab_tokens)}, 'pad': PAD, 'unk': UNK}, open(OUT + '/vocab.json', 'w'))
lens = lambda df: df['tokens'].map(len)
stats = {'seed': SEED, 'source': 'fancyzhx/yelp_polarity (Hugging Face parquet, revision bbf1c97a)',
         'source_rows': {'train': 560000, 'test': 38000}, 'dropped_empty_after_preprocessing': {'train': 26, 'test': 0},
         'rows': {'train': len(train_df), 'validation': len(val_df), 'test': len(test_df)},
         'class_balance': {k: df['label'].value_counts().sort_index().astype(int).to_dict() for k, df in (('train', train_df), ('validation', val_df), ('test', test_df))},
         'tokens_per_review': {k: {'mean': round(float(lens(df).mean()), 2), 'median': float(lens(df).median()), 'max': int(lens(df).max())} for k, df in (('train', train_df), ('validation', val_df), ('test', test_df))},
         'vocab_size': len(vocab_tokens), 'max_len': 180, 'stopwords_removed': len(STOPWORDS), 'stemming': USE_STEMMING,
         'test_labels_match_saved_predictions': bool(same)}
json.dump(stats, open(OUT + '/dataset_stats.json', 'w'), indent=2)
print(json.dumps(stats['tokens_per_review']))
