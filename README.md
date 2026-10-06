# DATA266 Lab 1 - Pair 41

Team members: Yuyao Ding (folder: `Yuyao_Ding`) and Pratiksha Kaushik (folder: `Pratiksha_Kaushik`).

Yuyao's recorded experiments, saved models and evidence for all three tasks have
been imported into the shared repository. Pratiksha's completed work for all three
tasks is in her `Pratiksha_Kaushik/` folders; see
[Pratiksha's work and artifact downloads](#pratikshas-work-and-artifact-downloads).
The [combined team report PDF](report/DATA266_Lab1_Report_Team_41.pdf) is in `report/`, alongside each member's supporting write-ups.

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

## Yuyao's artifact downloads

Model weights and processed data are stored in Google Drive. A Git clone contains
the code, configs, reports and recorded evidence; download the artifacts below
before running saved-model demos or reproducing a run.

| Task | Processed data | Checkpoints | Restore under / task guide |
| --- | --- | --- | --- |
| Task 1 | [Download](https://drive.google.com/drive/folders/1EfWk8oRTRdwjsv6qLH9zMbljhpX-Dmqz?usp=sharing) | [Download](https://drive.google.com/drive/folders/1Aq5tUVV39Tkx4q85AOq1TgamUxZ0IHev?usp=sharing) | [task1_llm/Yuyao_Ding/](task1_llm/Yuyao_Ding/README.md) |
| Task 2 | [Download](https://drive.google.com/drive/folders/1jzNHg-lIQ1ok9_8g7d_9rF2HsN9JOaO_?usp=sharing) | [Download](https://drive.google.com/drive/folders/18AzMJ8NbMmE7l38yGKZziBppHmq4LlIZ?usp=sharing) | [task2_sentiment/Yuyao_Ding/](task2_sentiment/Yuyao_Ding/README.md) |
| Task 3 | [Download](https://drive.google.com/drive/folders/1KWZo-O5WC11xQQ2eOS9T4UH_z6ZB2Upb?usp=sharing) | [Download](https://drive.google.com/drive/folders/1fkqIV4PLRg7x7v6cEMiWDrrx5O2ATTAV?usp=sharing) | [task3_gan/Yuyao_Ding/](task3_gan/Yuyao_Ding/README.md) |

Copy each Drive folder's contents into the corresponding `data_processed/` or
`checkpoints/` directory under the task's `Yuyao_Ding/` folder. Preserve run IDs,
filenames and JSON bytes, and avoid nesting the folder twice. Each destination
contains a tracked README with the expected layout and required files. Use an
account with read access; ask the folder owner if access is restricted.
Raw datasets and Task 3 metric-network downloads are described separately below
and in the task guides.

## Pratiksha's work and artifact downloads

Pratiksha Kaushik trained all three tasks independently in her own notebooks, with
her own data splits and seeds. Each task folder has a README with the folder layout,
how to reproduce the run, and every required metric.

| Task | Models and result | Hardware | Guide |
| --- | --- | --- | --- |
| 1: character GPT | 8 layers, 8 heads, 512-d, context 512 (25.56M params), 10 epochs; val CE 0.4742, perplexity 1.607, 0.684 bits/char, top-1 accuracy 84.7% | NVIDIA A100 40GB (Colab) | [task1_llm/Pratiksha_Kaushik/](task1_llm/Pratiksha_Kaushik/README.md) |
| 2: Yelp Polarity | Mean-embedding baseline, CNN (kernels 3/5/7) and BiGRU; 5 epochs each. Best: CNN, accuracy 0.9054, MCC 0.8108, significantly better than the baseline (McNemar p = 0.0006) | NVIDIA Tesla T4 (Colab) | [task2_sentiment/Pratiksha_Kaushik/](task2_sentiment/Pratiksha_Kaushik/README.md) |
| 3: CycleGAN | ResNet generators + 2-scale PatchGAN discriminators, 80 epochs; course score 50.2351 (FID 100.06, MiFID 0.4135), Kaggle public score −50.2351, human audit 3.49 / 5 (2 raters, 30 samples) | NVIDIA RTX 4090 | [task3_gan/Pratiksha_Kaushik/](task3_gan/Pratiksha_Kaushik/README.md) |

Model weights and large files are stored in Google Drive. Put checkpoint files into
the task's `checkpoints/` folder under `Pratiksha_Kaushik/`.

| Task | All files | Checkpoints |
| --- | --- | --- |
| Task 1 | [Download](https://drive.google.com/drive/folders/18IwhldDiOowviFXWqD22ZC2dRBwvGPEF?usp=drive_link) | [Download](https://drive.google.com/drive/folders/1XkTzqopbapev8RqSWLXST-E_gip88QYR?usp=drive_link) |
| Task 2 | [Download](https://drive.google.com/drive/folders/1l_VToNCca70yGB9zTINZdaeYycaoTFMZ?usp=drive_link) | [Download](https://drive.google.com/drive/folders/1sBI2vdiaSnv6S1KcWrIeYHQg5oVAT_k4?usp=drive_link) |
| Task 3 | [Download](https://drive.google.com/drive/folders/1xVVO0-Xpt1kmzFla6CumSy5gx788-WTC?usp=drive_link) | [Download](https://drive.google.com/drive/folders/1Ce8xc2Tt_9E9ZXQC9ou5aPAu1iLHpdq2?usp=drive_link) |

The Task 3 prediction folders (`pred_A2B/`, `pred_B2A/`) and figures have their own
Drive links in the READMEs inside those folders.

Other evidence:

- Reports: [Task 1](report/Pratiksha_Kaushik/Task1_Pratiksha_Kaushik.md), [Task 2](report/Pratiksha_Kaushik/Task2_Pratiksha_Kaushik.md), [Task 3](report/Pratiksha_Kaushik/Task3_Pratiksha_Kaushik.md).
- Raw logs and run manifests for all three tasks:
  [reproducibility/manifests/Pratiksha_Kaushik/](reproducibility/manifests/Pratiksha_Kaushik/README.md).
- To reproduce a run, open the task's final notebook in Colab on a GPU and run all cells.
  The notebooks download TinyStories and Yelp Polarity from Hugging Face. Task 3 uses
  the Kaggle Monet/photo data listed under "Data and full reproduction" below.

## Layout

- `task1_llm/{Pratiksha_Kaushik,Yuyao_Ding}/`: character-level GPT on TinyStories.
- `task2_sentiment/{Pratiksha_Kaushik,Yuyao_Ding}/`: three sentiment models per member.
- `task3_gan/{Pratiksha_Kaushik,Yuyao_Ding}/`: independently trained CycleGAN and bidirectional outputs.
- Each task has a shared `data/` folder for raw data.
- `reproducibility/raw_logs/<member>/`: original, unedited training logs.
- `reproducibility/manifests/<member>/`: environments, hardware and checkpoint-to-result mappings.
- `report/`: the [combined team report PDF](report/DATA266_Lab1_Report_Team_41.pdf) and supporting contributions in `Yuyao_Ding/` and `Pratiksha_Kaushik/`.

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
evaluator. The unchanged course notebook is tracked in Git; cached Inception/AlexNet
metric weights are excluded and can be downloaded by the evaluator as described in
the Task 3 guide. Training the GAN does not use pretrained model weights.

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

Git tracks the source notebooks and helpers, configs, requirements, original
logs/manifests, formal predictions, curves, audit materials, Kaggle CSV and the
three report contributions. Restore preprocessing records, formal best/last
checkpoints and the old quick-baseline weights through the Drive links above.
Task 3's archived quick-baseline code and recorded results are tracked in Git. Task 1's
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

See the [report directory](report/README.md) for the final PDF and each member's supporting write-ups. Follow the course instructions for the final submission.

Use relative paths and configuration files, and preserve raw training logs unchanged.
Do not commit credentials or personal absolute paths. Large data, preprocessing
arrays and checkpoint directories are excluded from ordinary Git commits by the
existing `.gitignore`; keep them accessible through documented shared storage.
Local presence does not mean these artifacts have been uploaded to GitHub.
