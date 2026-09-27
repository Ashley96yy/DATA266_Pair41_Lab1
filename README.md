# DATA266 Lab 1 - Pair 41

Team members: Yuyao Ding (folder: `Yuyao_Ding`) and Pratiksha Kaushik (folder: `Pratiksha_Kaushik`).

Yuyao's recorded experiments, saved models and evidence for all three tasks have
been imported into the shared repository. Pratiksha's existing folders are preserved;
her completion status has not been verified in this update. The combined final
team report remains pending.

Each member independently completes all three tasks. Task 2 requires one baseline
and two experimental models per member. Raw datasets may be shared; preprocessing
outputs and model work belong to each member.

## RTX 4090 work and consolidated submission

Yuyao completed the new Task 1-3 experiments on a separate Windows computer with
an NVIDIA GeForce RTX 4090, then transferred the code, recorded results and
supporting evidence back to this shared repository. These materials are therefore
being committed and pushed together in one consolidated update after the transfer.
The Git commit time reflects repository integration; the original run IDs,
timestamps, raw logs and manifests document when the experiments were performed.

| Task | Recorded result | Guide |
| --- | --- | --- |
| 1: character GPT | 10 epochs; best epoch 10 | [Setup, demo and results](task1_llm/Yuyao_Ding/README.md) |
| 2: Yelp Polarity | Baseline, TextCNN and BiLSTM; five epochs each | [Setup, demo and results](task2_sentiment/Yuyao_Ding/README.md) |
| 3: CycleGAN | 100 x 1,000 updates; best epoch 70 | [Setup, demo and results](task3_gan/Yuyao_Ding/README.md) |

## Layout

- `task1_llm/{Pratiksha_Kaushik,Yuyao_Ding}/`: character-level GPT on TinyStories.
- `task2_sentiment/{Pratiksha_Kaushik,Yuyao_Ding}/`: three sentiment models per member.
- `task3_gan/{Pratiksha_Kaushik,Yuyao_Ding}/`: independently trained CycleGAN and bidirectional outputs.
- Each task has a shared `data/` folder for raw data.
- `reproducibility/raw_logs/<member>/`: original, unedited training logs.
- `reproducibility/manifests/<member>/`: environments, hardware and checkpoint-to-result mappings.
- `report/`: individual contributions and the combined team report checklist.

## Start here

Work from the repository root and keep the relative paths intact. Restore any
missing model or data files from the resource package or documented shared storage.
Use Python 3.12 and create a fresh environment on each computer. Opening a saved
notebook displays its recorded outputs; **Run All starts another training run**.
Use the demo commands below to inspect the trained models without training.

Mac (Apple Silicon) or Linux CPU, from this project directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r task1_llm/Yuyao_Ding/requirements.txt -r task2_sentiment/Yuyao_Ding/requirements.txt -r task3_gan/Yuyao_Ding/requirements.txt
.venv/bin/python -m ipykernel install --user --name python3 --display-name "DATA266 Python"
```

Windows with an NVIDIA GPU (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r task1_llm/Yuyao_Ding/requirements.txt -r task2_sentiment/Yuyao_Ding/requirements.txt -r task3_gan/Yuyao_Ding/requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name python3 --display-name "DATA266 Python"
```

For Linux NVIDIA, install the same CUDA wheels before the requirements using
`.venv/bin/python`. For Windows/Linux CPU-only, use the `cpu` wheel index instead
of `cu128`. The per-task guides describe device selection. The installed runtime
versions from the recorded experiments are preserved in the run manifests.

## Saved-model demos

Mac/Linux:

```bash
.venv/bin/python task1_llm/Yuyao_Ding/src/demo.py --device cpu --prompt "Once upon a time,"
.venv/bin/python task2_sentiment/Yuyao_Ding/src/demo.py --model all --device cpu --text "The food was good, but the service was not."
.venv/bin/python task3_gan/Yuyao_Ding/src/demo.py --image task3_gan/data/photo_jpg/011f46de73.jpg --direction B2A --output monet_demo.jpg
.venv/bin/python task3_gan/Yuyao_Ding/src/demo.py --image task3_gan/data/monet_jpg/0260d15306.jpg --direction A2B --output photo_demo.jpg
```

Windows: use `.\.venv\Scripts\python` in place of `.venv/bin/python` with the same
arguments. Choose a new output image filename when repeating a Task 3 demo.
These demos need neither the full datasets nor a running Jupyter server.

To inspect notebooks on Mac/Linux, run `.venv/bin/python -m jupyter lab`; on
Windows use `.\.venv\Scripts\python -m jupyter lab`. Select the environment above.

## Data and full reproduction

The imported resource package excludes full datasets and large regenerated
preprocessing arrays. This local checkout retains its existing TinyStories raw
files and Task 1 preprocessing arrays. Only one Monet and one photo input were
imported for the demos: **these two images are not the training dataset**. Restore
any missing complete datasets before smoke checks, full evaluation or a new run.

| Data | Download and destination |
| --- | --- |
| TinyStories | [Task 1 Drive folder](https://drive.google.com/drive/folders/14kpF6QaLSHfF4Ru-T6KEpcctCpRqDX9x): extract `TinyStories-train.txt` and `TinyStories-valid.txt` directly into `task1_llm/data/`. Use the original files, not V2-GPT4. |
| Yelp Polarity | [Team Drive folder](https://drive.google.com/drive/folders/1Enmnu7482fsa0-9p_2rmzsyEImcwwMSg), Task 2 data: extract `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` into `task2_sentiment/data/`. |
| Monet/photo | [Task 3 Drive folder](https://drive.google.com/drive/u/1/folders/1z9cw21bh1u6PMAjUWG_wiAGlcRmHjxO5): extract all 300 Monet and 7,038 photo JPEGs into `task3_gan/data/monet_jpg/` and `photo_jpg/`, without an extra `dataset/` parent. |

Use an account with read access. The per-task guides and manifests record expected
files, hashes, preprocessing, splits and seeds. Keep saved vocabulary/split JSON
files byte-for-byte unchanged. Regenerate omitted arrays using the documented
preprocessing steps. The optional `real_stats.npz` is not included or used by our
evaluator; the unchanged course notebook and cached Inception/AlexNet metric weights
are included. Training the GAN does not use pretrained model weights.

After setup and restoring TinyStories, this single command prepares missing
arrays and runs the Task 1 smoke test without starting ten-epoch training:

```bash
.venv/bin/python task1_llm/Yuyao_Ding/src/smoke_test.py
```

Windows:

```powershell
.\.venv\Scripts\python task1_llm/Yuyao_Ding/src/smoke_test.py
```

The other smoke commands and full training/evaluation instructions are in their
task guides. Checkpoints can reproduce inference; new training on another device
need not produce bitwise-identical weights or metrics. Physical Mac/Linux/Colab
execution remains to be checked. Current pipeline validation was on Windows CPU/CUDA.

## Contents and evidence

Included are the source notebooks and helpers, configs, requirements, small
preprocessing records, five formal best checkpoints, formal last states, Task 3's
old quick-baseline best/code/results, original logs/manifests, formal predictions,
curves, audit materials, Kaggle CSV and the three report contributions. Task 1's
epoch-10 file is both best and last. The three Task 2 `best_epoch_*` files named
inside `last.pt` are retained alongside `best.pt` so those references still resolve.
The completed Task 3 training budget cannot be extended simply by resuming `last.pt`.

Checkpoint folders retain their original run IDs. The table below identifies the
saved epochs; the checkpoint metadata records them too. Task 3 also includes
`best.json` and `last.json` beside the formal weights.

| Model | Best checkpoint (epoch) | Last checkpoint (epoch) |
| --- | --- | --- |
| Task 1: character GPT | `epoch_010.pt` (10) | Same file (10) |
| Task 2: Baseline | `best.pt` (5) | `last.pt` (5) |
| Task 2: TextCNN | `best.pt` (3) | `last.pt` (5) |
| Task 2: BiLSTM | `best.pt` (3) | `last.pt` (5) |
| Task 3: formal CycleGAN | `best.pt` (70) | `last.pt` (100) |
| Task 3: old quick baseline | `best.pt` (39) | Not included |

Historical logs and validation manifests are preserved as records. The import
also preserves this checkout's original Mac Task 1 checkpoints, smoke outputs,
raw data and preprocessing arrays. The Mac run remains available for comparison;
the current Task 1 reports use the RTX 4090 run.

Use the [report checklist](report/README.md) when combining the individual sections.
Real human ratings, Kaggle scores/rank, teammate comparison and the final PDF
remain pending. Follow the course instructions for the final submission.

Use relative paths and configuration files, and preserve raw training logs unchanged.
Do not commit credentials or personal absolute paths. Large data, preprocessing
arrays and checkpoint directories are excluded from ordinary Git commits by the
existing `.gitignore`; keep them accessible through documented shared storage.
Local presence does not mean these artifacts have been uploaded to GitHub.
