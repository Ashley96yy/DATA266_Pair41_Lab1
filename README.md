# DATA266 Lab 1 — Pair 41

Team members: Yuyao Ding (folder: `Yuyao_Ding`) and Pratiksha Kaushik (folder: `Pratiksha_Kaushik`).

## Status

Repository scaffold only. No models have been implemented or trained, and no metrics or reproduction commands have been verified yet.

Each member independently completes all three tasks. Task 2 requires one baseline and two experimental models per member. Raw datasets may be shared; preprocessing outputs and model work belong to each member.

## Layout

- `task1_llm/{Pratiksha_Kaushik,Yuyao_Ding}/`: character-level GPT on TinyStories.
- `task2_sentiment/{Pratiksha_Kaushik,Yuyao_Ding}/`: three sentiment models per member.
- `task3_gan/{Pratiksha_Kaushik,Yuyao_Ding}/`: independently trained CycleGAN and bidirectional outputs.
- Each task has a shared `data/` folder for raw data.
- `reproducibility/raw_logs/<member>/`: original, unedited training logs.
- `reproducibility/manifests/<member>/`: environments, package versions, hardware and checkpoint-to-result mappings.
- `report/`: combined team report and report checklist.

## Setup and reproduction — pending implementation

Record actual dependency versions and environment setup after implementation. Document how to reproduce each member's runs and locate their results. Before submission, provide a single verified command that runs a smoke test for at least one member. No working command is available yet.

Use relative paths and configuration files. Never commit credentials or personal absolute paths. Preserve raw training logs unchanged. Back up checkpoints and data outside GPU lab machines; large binary artifacts are excluded from ordinary Git commits by default and require a documented storage/retrieval plan.

## Before training

- Confirm deadline and team information on Canvas.
- Confirm Task 2 dataset: the task text says Yelp Polarity, while the example tree says IMDB.
- Confirm the unit of Task 1's 100K/10K split and the scope of Task 2's 20-error review.
- Agree on common evaluation protocols and distinguish each member's architecture/hyperparameter choices.
- Confirm Kaggle submission format and the 30-sample, two-rater blind audit protocol.
- Review the course's AI-use rule; core architecture decisions and analysis must reflect each member's own understanding.
- Reserve GPU time and arrange regular backups.
