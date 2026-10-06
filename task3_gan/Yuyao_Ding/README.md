# Task 3 - Yuyao Ding

This combined archive uses an environment inside the extracted project. Start with the [project setup](../../README.md). In older Windows examples below, replace `..\.venv\Scripts\python` with `.\.venv\Scripts\python`. The original computer kept its environment one directory above the project.


The nine-block FP32 CycleGAN completed 100 epochs and 100,000 updates on RTX 4090. Validation selected epoch 70; formal evaluation and the 30-case two-rater audit are complete. Yuyao's Kaggle Public score is -52.8347. See [results.md](results.md), [the eight-case analysis](failure_analysis.md), [the report contribution](../../report/Yuyao_Ding/Task3_Yuyao_Ding.md) and [the combined team report](../../report/DATA266_Lab1_Report_Team_41.pdf).

The original six-block baseline remains available, with its code/configuration in [quick_6block.zip](baselines/quick_6block.zip). Its saved checkpoints, outputs and raw logs are preserved.

## Download processed data and checkpoints

These artifacts are stored in Google Drive and are not included in a Git clone.
Restore the folder contents to the following paths relative to the repository root:

| Artifact | Google Drive | Restore destination | Details |
| --- | --- | --- | --- |
| Processed data | [Download folder](https://drive.google.com/drive/folders/1KWZo-O5WC11xQQ2eOS9T4UH_z6ZB2Upb?usp=sharing) | `task3_gan/Yuyao_Ding/data_processed/` | [Layout and required files](data_processed/README.md) |
| Checkpoints | [Download folder](https://drive.google.com/drive/folders/1fkqIV4PLRg7x7v6cEMiWDrrx5O2ATTAV?usp=sharing) | `task3_gan/Yuyao_Ding/checkpoints/` | [Run IDs and model selection](checkpoints/README.md) |

Preserve nested run directories and file bytes; avoid an extra directory level
when extracting a downloaded folder. Use an account with read access. These links
are for Yuyao's artifacts; raw datasets are listed separately in the data section.

## Files

- `src/task3_gan.ipynb`: data splits, losses, training, validation, checkpoints and plots.
- `src/cyclegan.py`: the generator/discriminator definitions and shared image/checkpoint helpers.
- `src/evaluate.py`: full metrics, generated images, the course CSV and the human audit.
- `src/demo.py`: translate a new image using a saved checkpoint.
- `src/smoke_test.py`: execute a small copy of the notebook without editing the original.
- `configs/training.json`: formal experiment settings.

The unchanged course notebook is at `../Part3_Evaluation_Script.ipynb`. Our evaluator uses its Inception preprocessing, FID and MiFID definitions. It removes the obsolete `sqrtm(..., disp=False)` call so the calculation works with current SciPy.

## Saved results

Current run: `task3_formal_20260927T075834Z_d335ae80`, selected epoch 70.

- [Executed notebook](outputs/task3_formal_20260927T075834Z_d335ae80/training_notebook.ipynb) and [training curves](outputs/task3_formal_20260927T075834Z_d335ae80/training_curves.png).
- [Selected validation examples](outputs/task3_formal_20260927T075834Z_d335ae80/validation/epoch_070/input_translation_cycle.png): input, translation, cycle from left to right.
- [Full metric table](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/full_metrics_report.csv) and [evaluation record](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/evaluation.json).
- [Kaggle CSV](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/submission.csv): local FID 105.250404 and MiFID 0.419085, averaged across directions. Yuyao's uploaded `submission.csv` received Public score **-52.8347** ([Kaggle screenshot](outputs/kaggle_public_submissions.png)); private score and personal rank are not shown.
- [Completed 30-case audit](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/audit_results.json): mean style 2.67/5, content 2.90/5, artifact-free 2.48/5; exact agreement and Cohen's kappa are in [results.md](results.md).
- [Best weights](checkpoints/task3_formal_20260927T075834Z_d335ae80/best.pt) and [checksum sidecar](checkpoints/task3_formal_20260927T075834Z_d335ae80/best.json).

Open the executed notebook to inspect the saved run. Use the demo below for inference; rerunning all training cells starts another experiment.

## Environment

Use Python 3.12. Run the following commands from the repository root, which contains `task1_llm`, `task2_sentiment` and `task3_gan`. Do not transfer a Windows virtual environment to a Mac or Linux machine; create a new one there.

Windows with an NVIDIA GPU:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r task3_gan/Yuyao_Ding/requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name python3 --display-name "DATA266 Python"
```

Apple Silicon Mac:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r task3_gan/Yuyao_Ding/requirements.txt
.venv/bin/python -m ipykernel install --user --name python3 --display-name "DATA266 Python"
```

Linux with an NVIDIA GPU:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r task3_gan/Yuyao_Ding/requirements.txt
.venv/bin/python -m ipykernel install --user --name python3 --display-name "DATA266 Python"
```

For Windows/Linux CPU-only use, install the same torch and torchvision versions with `--index-url https://download.pytorch.org/whl/cpu` before installing the requirements. The Mac command above targets Apple Silicon; Intel Mac package compatibility has not been tested.

The notebook selects CUDA, then MPS, then CPU. Set `DEVICE = "cpu"` to override it. The formal configuration uses float32 on all devices (`cuda_bfloat16 = false`). Data loaders use zero workers so notebook execution does not depend on multiprocessing behavior. Evaluation uses CPU for Inception/LPIPS on a Mac with MPS. Physical Mac, Linux and Colab checks remain pending.

The first evaluation downloads official Inception-v3 and AlexNet weights, about 337 MiB in total, into `task3_gan/data/metric_weights/`. These are metric networks only. All CycleGAN networks are initialized from scratch. After the weights are cached, training and evaluation can run offline.

On Colab, install the requirements in a setup cell, restart the runtime if torch was changed, mount the complete project with `USE_COLAB_DRIVE = True`, and set `PROJECT_ROOT` to its Drive path. Keep this false for local runs. A mounted Drive is convenient but slower for many small files; a local Colab copy needs its new checkpoints and results copied back before the runtime ends.

## Run the saved model

From the repository root, after setup, this loads the current best checkpoint without training or downloading metric weights:

```powershell
.\.venv\Scripts\python task3_gan/Yuyao_Ding/src/demo.py --image task3_gan/data/photo_jpg/011f46de73.jpg --direction B2A --output monet_demo.jpg
```

```bash
.venv/bin/python task3_gan/Yuyao_Ding/src/demo.py --image task3_gan/data/photo_jpg/011f46de73.jpg --direction B2A --output monet_demo.jpg
```

The demo defaults to CPU. For Monet -> Photo, use a Monet input such as `task3_gan/data/monet_jpg/0260d15306.jpg` and `--direction A2B`. Use a different output filename on each invocation. Any readable input image can be supplied; it is resized to 256 x 256. To select this exact model explicitly, add `--checkpoint task3_gan/Yuyao_Ding/checkpoints/task3_formal_20260927T075834Z_d335ae80/best.pt`.

On the existing Windows installation, replace `.\.venv\Scripts\python` with `..\.venv\Scripts\python`, since its environment is one directory above the repository. On another machine, create its environment using the setup section. Windows-written checkpoint pointers are accepted on all three operating systems; actual execution has been checked on Windows CPU/CUDA only.

## Data

Download `dataset.zip` from the [team Drive data folder](https://drive.google.com/drive/u/1/folders/1z9cw21bh1u6PMAjUWG_wiAGlcRmHjxO5). Use an account with read access. Extract only `dataset/monet_jpg` and `dataset/photo_jpg` into `task3_gan/data/`, without retaining the extra `dataset` parent. Skip `.DS_Store`, `._*` and `__MACOSX` metadata. Expected counts are 300 Monet images and 7,038 photographs, all 256 x 256 RGB JPEGs. Keep the original archive as a backup.

The [inventory](../../reproducibility/manifests/Yuyao_Ding/task3_data_inventory.json) has the source links and SHA-256 of every image. Training verifies these hashes. The nine duplicate-photo groups stay intact in the raw data and are assigned together to one partition. No original images are removed.

`data_processed/split_seed41.json` records the shared split for Yuyao's runs. It uses about 80/10/10 percent train/validation/test in each domain, grouping identical image files. Changing fractions or source data requires a separately named split/seed. Smoke subsets do not overwrite this full split. As in the official unaligned loader, domain A follows shuffled indices modulo its size and domain B is randomly sampled with replacement. All 240 training Monet images and 5,631 training photos remain available each epoch. With the 1,000-update limit, Monet images are used about 4.17 times each on average, without guaranteeing complete coverage in every epoch. Sampling continues across epochs; no fixed 1,000-photo subset is created.

The optional `real_stats.npz` is not included in this package. The supplied course evaluation notebook does not read it; neither does our matching evaluator.

## Small check

From the repository root:

```powershell
.\.venv\Scripts\python task3_gan/Yuyao_Ding/src/smoke_test.py
```

```bash
.venv/bin/python task3_gan/Yuyao_Ding/src/smoke_test.py
```

Add `--device cpu` to force CPU. Add `--check-resume` to compare interrupted/resumed and uninterrupted training. The normal check runs two epochs of two updates with a small 64 x 64 network, then exercises both evaluation protocols, export, cycle examples and blank audit forms. It saves an executed notebook under `outputs/smoke_checks/`. It leaves the source notebook, formal metrics and latest-formal pointer unchanged. Smoke scores are not model-quality results.

**This machine already has the environment one directory above the repository.** Use `..\.venv\Scripts\python` here instead of `.\.venv\Scripts\python`; no reinstall is needed.

## Formal training

On this Windows machine, from the repository root:

```powershell
..\.venv\Scripts\python -m jupyter lab task3_gan/Yuyao_Ding/src/task3_gan.ipynb
```

Reload the notebook from disk and restart the kernel. The parameters are ready for all 100 epochs: `RUN_FORMAL_TRAINING = True`, `RESUME_CHECKPOINT = None` and `STOP_AFTER_EPOCH = None`; run all cells to start. Set `STOP_AFTER_EPOCH` to an epoch number only for a planned pause, or `RUN_FORMAL_TRAINING = False` for a smoke run. `RUN_EVALUATION = True` automatically evaluates the selected model after the training budget finishes. Running all cells again with no resume path starts a new run.

Formal defaults: two nine-block ResNet generators with 64 base channels and transpose-convolution upsampling; two 70 x 70 PatchGAN discriminators; instance normalization without affine parameters or running statistics; no dropout; normal weight initialization with standard deviation 0.02. There are 28,285,832 parameters across all four networks.

Training uses least-squares GAN loss, cycle weight 10 and identity weight 5 (equivalent to the official `lambda_identity=0.5` multiplied by cycle weight 10). Adam uses learning rate 0.0002 with betas (0.5, 0.999), batch size 1 and a replay pool of 50. Images are resized to 286 x 286, randomly cropped to 256 x 256, horizontally flipped and normalized to [-1, 1]. These settings follow the [authors' implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

The budget is 100 epochs of 1,000 updates, or 100,000 updates in total: 50 epochs at the initial learning rate followed by 50 epochs of linear decay. Training uses FP32. This reduces the compute budget while keeping the architecture, losses and unpaired sampling unchanged. It is not the authors' 200-full-epoch protocol. The course specifies no minimum epoch count for Task 3; assess the result from validation, final metrics and images.

The lowest mean bidirectional validation KID selects the checkpoint. Validation uses the same fixed 30 images from each domain after every epoch. Also inspect the saved previews and loss/gradient curves: stable losses alone do not establish good translations. Quality and convergence must be assessed after training.

The completed run recorded 18,602.59 seconds of training wall time (about 5 h 10 min), plus 41.54 seconds of timed final evaluation. It processed 5.46 updates/second during training loops and peaked at 2,596.36 MiB of allocated CUDA tensors. These are measured values for this RTX 4090 run, not expected performance on another machine.

[Alignment checks](../../reproducibility/manifests/Yuyao_Ding/task3_official_settings_checks.json) record the earlier 200-full-epoch configuration: the networks matched the official classes with identical weights, full-size FP32 CUDA updates passed, and both new and original checkpoints ran in a fresh CPU demo process. The [budget checks](../../reproducibility/manifests/Yuyao_Ding/task3_budget_checks.json) record the shorter schedule and preceding checkpoint implementation. The [checkpoint checks](../../reproducibility/manifests/Yuyao_Ding/task3_best_checkpoint_checks.json) verify immediate best saves, recovery after a newer best overwrites the file, CUDA smoke evaluation and exact resume. These checks passed on Windows CPU/CUDA; physical Mac/Linux tests remain pending.

Validation still runs every epoch. Whenever its mean KID improves, `best.pt` is overwritten immediately; no best-model copy remains in memory between saves. Every five epochs, the notebook updates the full `last.pt`, which also includes the selected best checkpoint at that point. The final epoch and a planned `STOP_AFTER_EPOCH` force a full save. Each run keeps two checkpoint files, `best.pt` and `last.pt`, with their JSON sidecars. An unexpected interruption can lose up to five epochs of training progress since `last.pt`, while the latest saved `best.pt` remains available for inference.

To resume, keep the same configuration and set `RESUME_CHECKPOINT` to a saved `last.pt`. Resume starts after the last fully saved epoch, restores both optimizers, replay pools, random states and that epoch's best selection, and creates a new run directory. Recovery reads the best snapshot inside `last.pt`, so a later improvement to the separate `best.pt` cannot change the restored selection. Move the checkpoint files with their matching JSON sidecars when transferring a run. Exact repeatability was checked on the same Windows/CUDA setup; moving across devices need not be bitwise identical.

`latest_formal.json` now selects the completed nine-block run. The earlier six-block checkpoint still loads with the matching compatibility branch; specify its path explicitly for comparison. A new training experiment starts from scratch, while resuming requires an unfinished `last.pt` and the same configuration. The completed 100-epoch budget cannot be extended by treating `best.pt` as a recovery checkpoint. The quick-baseline ZIP preserves the original source/configuration and is not a second daily-use notebook.

## Evaluation and demo

Formal evaluation is already saved at the links above. To generate a separate evaluation directory from the current best checkpoint, run:

```powershell
..\.venv\Scripts\python task3_gan/Yuyao_Ding/src/evaluate.py --device cuda
```

```bash
.venv/bin/python task3_gan/Yuyao_Ding/src/evaluate.py --device cpu
```

`A2B` means Monet → Photo; `B2A` means Photo → Monet. Add `--checkpoint path/to/best.pt` to select another saved run. `demo.py` defaults to CPU and only needs the model source, checkpoint and its matching JSON sidecar, plus the input image. Inputs are resized to the configured square image size. It refuses to overwrite an existing output file.

The evaluator writes a new evaluation directory each time:

- `heldout_test/`: a fixed balanced subset (30 per domain for these data), generated images, cycle examples and all model-quality metrics. No training or checkpoint selection uses these images. The small Monet test set makes distribution metrics noisy.
- `course_kaggle/`: the first 300 sorted source and real-target filenames per domain, matching the provided notebook. These images include training data, so this is reported separately from held-out generalization.
- `submission.csv`: one row with `ID=1`, `FID`, `MiFID`; each score averages both directions. Only this CSV is the Kaggle submission. No automated upload occurs.
- `evaluation.json`, `full_metrics_report.csv`: metrics, configurations, hashes, metric-network versions, resource costs and evidence paths.
- `audit/`: fixed anonymous samples, instructions, and two completed rater CSVs for this run. `audit_key.json` is outside this folder and should not be shown to raters. `audit_results.json` records the aggregate scores and agreement.

To rebuild the metric CSVs from an existing `evaluation.json`, without loading a model or rerunning inference:

```powershell
$task3Evaluation = "task3_gan/Yuyao_Ding/outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de"
..\.venv\Scripts\python task3_gan/Yuyao_Ding/src/evaluate.py --export-only --evaluation-dir $task3Evaluation
```

```bash
task3_evaluation="task3_gan/Yuyao_Ding/outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de"
.venv/bin/python task3_gan/Yuyao_Ding/src/evaluate.py --export-only --evaluation-dir "$task3_evaluation"
```

The table includes loss and gradient-norm means for each training epoch (`epoch_001`, etc.). These are online training averages, so their `checkpoint_id` is blank; they are not evaluations of the selected checkpoint. Training cycle/identity losses use the normalized [-1, 1] image range before loss weights, while `G_total` includes those weights. The latest formal evaluation also updates the member-level CSV copies. Re-exporting a historical or smoke evaluation leaves those copies alone, and completed human scores are retained. Normal evaluation and audit aggregation use the same export function.

KID is unbiased polynomial MMD on its raw scale and can be negative. Its subset standard deviation is not a confidence interval. Generative precision/recall and density/coverage use 5-nearest-neighbour feature radii. Cycle L1 is reported in pixel range [0,1]. LPIPS uses AlexNet v0.1 on input vs translation and input vs cycle; the task has no paired target image. Content preservation uses Inception cosine between each input and its exported JPEG. The course MiFID is sorted, index-matched cosine distance, not a nearest-neighbour memorization penalty.

FID uses 2,048-dimensional features with fewer images than dimensions, so its covariance matrices are rank deficient. SciPy may print a singular-matrix warning even when the result is finite. The evaluator retains the course calculation and checks finiteness; this warning alone is not a failed training run.

The required audit is complete for this formal evaluation: Liming Jiang and Sherry Tang each rated the same 30 fixed anonymous cases (15 per direction) on 1-5 scales for style, content and artifact-free quality. The [instructions](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/audit/instructions.md) required independent ratings. The two completed CSVs and [aggregate results](outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de/audit_results.json) are saved with the evaluation. To recompute the aggregate from these forms in an environment with the project dependencies, run:

```powershell
$task3Evaluation = "task3_gan/Yuyao_Ding/outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de"
..\.venv\Scripts\python task3_gan/Yuyao_Ding/src/evaluate.py --evaluation-dir $task3Evaluation --rater1 "$task3Evaluation/audit/rater_1.csv" --rater2 "$task3Evaluation/audit/rater_2.csv"
```

```bash
task3_evaluation="task3_gan/Yuyao_Ding/outputs/task3_formal_20260927T075834Z_d335ae80/evaluation_20260927T130841Z_a267de"
.venv/bin/python task3_gan/Yuyao_Ding/src/evaluate.py --evaluation-dir "$task3_evaluation" --rater1 "$task3_evaluation/audit/rater_1.csv" --rater2 "$task3_evaluation/audit/rater_2.csv"
```

This writes mean scores, exact agreement and Cohen's kappa overall and by direction, and adds them to that evaluation's metric table. If both raters give one identical constant score, kappa is undefined; agreement is still reported. The saved audit has complete scores for both raters.

## Saved evidence and reproduction

Training writes `checkpoints/<run>/`, `outputs/<run>/`, `reproducibility/raw_logs/Yuyao_Ding/<run>/` and `reproducibility/manifests/Yuyao_Ding/<run>/`. Manifests freeze sources, package versions, data splits, configuration, hardware and checkpoint hashes. Resuming uses new raw-log files and identifies the parent run. Each checkpoint has a matching `.json` checksum record; copy both together.

The combined `Yuyao_Ding_Lab1_Resources.zip` contains all three tasks, including the updated metric tables and helpers, formal best/last states, old baseline best, source/configs, recorded outputs, logs, manifests, official evaluator and metric-network cache. The image folders contain only the two demo inputs; restore the complete dataset from Drive before training, smoke testing or full evaluation. The completed 100-epoch budget remains complete even though `last.pt` is included.

For a GitHub checkout, use the Drive links above to restore checkpoints and the
saved split. Keep checkpoint JSON sidecars for checksum verification. The optional
`SHA256SUMS.txt` belongs to the original resource archive, not this repository.

The original run snapshots remain unmodified. The current helpers add compatibility for Windows-written path records and their existing checkpoint source hashes, plus complete CSV exports; model definitions and metric formulas are unchanged. The [final validation record](../../reproducibility/manifests/Yuyao_Ding/task3_final_validation.json) records the preceding cold checkpoint demos, path compatibility and evidence checks.

Remaining work includes the comparison with Pratiksha's Task 3 implementation and results, and the final team PDF. Her Task 3 files are now in the repository; architectural and hyperparameter differences still need to be documented. Yuyao's private Kaggle score and individual rank are not available. The team rank of 32 reflects Pratiksha's selected submission, not Yuyao's personal result.

References: [CycleGAN paper/project](https://junyanz.github.io/CycleGAN/), [LPIPS authors' implementation](https://github.com/richzhang/PerceptualSimilarity), [precision/recall/density/coverage authors' implementation](https://github.com/clovaai/generative-evaluation-prdc), [PyTorch wheel compatibility](https://pytorch.org/get-started/previous-versions/).
