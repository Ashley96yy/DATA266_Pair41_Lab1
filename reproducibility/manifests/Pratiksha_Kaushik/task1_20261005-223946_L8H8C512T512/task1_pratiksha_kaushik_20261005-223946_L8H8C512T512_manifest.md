# Manifest — Task 1, pratiksha_kaushik, run 20261005-223946_L8H8C512T512

## Environment

| item | value |
|---|---|
| python | 3.13.15 |
| platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| torch | 2.11.0+cu130 |
| cuda | 13.0 |
| cudnn | 92700 |
| gpu | NVIDIA A100-SXM4-40GB |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| datasets | 4.8.5 |
| ftfy | 6.3.1 |
| amp_dtype | torch.bfloat16 |

Full package list (environment file): `reproducibility/manifests/task1_pratiksha_kaushik_20261005-223946_L8H8C512T512_requirements-lock.txt`  
Raw log: `reproducibility/raw_logs/task1_pratiksha_kaushik_20261005-223946_L8H8C512T512_runlog.txt`

## Checkpoint → result mapping

| file | epoch | step | val CE | used for | sha256 (first 16) |
|---|---|---|---|---|---|
| task1_llm/pratiksha_kaushik/checkpoints/ckpt_best.pt | 10 | 27360 | 0.4742 | results.md, metrics_report.csv, failure_analysis.md, outputs/samples/, outputs/human_audit.csv | 4d9d958862777fe0 |
| task1_llm/pratiksha_kaushik/checkpoints/ckpt_last.pt | 10 | 27360 | 0.4742 | resuming training | 59f2908ffb333e4b |

## Data

| item | value |
|---|---|
| name | roneneldan/TinyStories |
| source_split | train |
| split_seed | 1337 |
| n_train_stories | 100000 |
| n_val_stories | 10000 |
| split_indices_file | task1_llm/pratiksha_kaushik/data_processed/split_indices.json |
| vocab_file | task1_llm/pratiksha_kaushik/data_processed/vocab.json |
| vocab_size | 80 |
| train_chars | 89656644 |
| val_chars | 8991471 |
| train_unk_rate | 1.2715175910443403e-06 |
| val_unk_rate | 2.113113638469167e-06 |

## Configuration

| key | value |
|---|---|
| dataset_name | roneneldan/TinyStories |
| seed | 1337 |
| n_train_stories | 100000 |
| n_val_stories | 10000 |
| min_story_chars | 50 |
| min_char_freq | 20 |
| block_size | 512 |
| n_layer | 8 |
| n_head | 8 |
| n_embd | 512 |
| dropout | 0.1 |
| epochs | 10 |
| batch_size | 64 |
| peak_lr | 0.0008 |
| min_lr | 8e-05 |
| warmup_frac | 0.03 |
| weight_decay | 0.1 |
| beta1 | 0.9 |
| beta2 | 0.95 |
| grad_clip | 1.0 |
| spike_factor | 1.3 |
| log_every | 50 |
| eval_interval | 500 |
| eval_batches | 40 |
| final_train_eval_batches | None |
| keep_epoch_weights | False |
| compile | True |
| repo_root | /content/drive/MyDrive/team-repo |
| member_name | pratiksha_kaushik |