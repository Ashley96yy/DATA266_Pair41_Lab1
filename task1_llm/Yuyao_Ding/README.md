# Task 1 — Yuyao Ding

This combined archive uses an environment inside the extracted project. Start with the [project setup](../../README.md). In older Windows examples below, replace `..\.venv\Scripts\python` with `.\.venv\Scripts\python`. The original computer kept its environment one directory above the project.


The notebook is `src/task1_llm.ipynb`. The reported run is `train_20260927T000510Z_b9df2210`
(NVIDIA GeForce RTX 4090, 10 completed epochs). `results.md`, `failure_analysis.md`
and notebook Section 1.4 use this run. `metrics_report.csv` contains its 52 rows
first, followed by the 52 historical Mac rows, each identified by `model_id`.

## Environment

Install Python 3.12 first. Run these commands from the repository root (the
directory containing the top-level README and `task1_llm/`). Create a separate
environment on each computer; do not copy `.venv` between operating systems.

Windows with an NVIDIA GPU (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r task1_llm/Yuyao_Ding/requirements.txt
.\.venv\Scripts\python -m jupyterlab task1_llm/Yuyao_Ding/src/task1_llm.ipynb
```

Mac:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r task1_llm/Yuyao_Ding/requirements.txt
python -m jupyterlab task1_llm/Yuyao_Ding/src/task1_llm.ipynb
```

Linux with an NVIDIA GPU:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r task1_llm/Yuyao_Ding/requirements.txt
python -m jupyterlab task1_llm/Yuyao_Ding/src/task1_llm.ipynb
```

On Windows or Linux without an NVIDIA GPU, use the PyTorch `cpu` index instead
of `cu128` in the installation command. GPU use also requires a compatible NVIDIA
driver. If `python3.12 -m venv` is unavailable on Linux, install your distribution's
Python 3.12 venv package first. Mac uses the regular PyPI installation shown above.

Choose the Python kernel from this environment. PyTorch installation commands
follow the [official 2.8.0 instructions](https://pytorch.org/get-started/previous-versions/#v280).
`device: "auto"` in `configs/training.json` selects CUDA, then Apple MPS, then CPU.
The smoke test prints the device it actually uses. CPU, CUDA and MPS runs need
not give identical numerical results, even with the same seed.

Windows CPU and RTX 4090 have passed local tests. The revised code has not been
tested on a physical Mac or Linux machine. The tested Windows environment used
Python 3.12.14, torch 2.8.0+cu128, numpy 2.5.3, matplotlib 3.11.2, psutil 7.2.2,
ipykernel 6.31.0 and jupyterlab 4.6.4. Per-run manifests record the versions used
for that run; the requirements file allows compatible versions of the support libraries.

## Data and saved weights

Ordinary Git commits exclude raw data, processed arrays and checkpoints. The
separate trained-model sharing ZIP described below includes the best checkpoint
and vocabulary, but excludes raw text and encoded training arrays.

- Download `TinyStories.zip` from the team's
  [Task 1 data folder on Google Drive](https://drive.google.com/drive/folders/14kpF6QaLSHfF4Ru-T6KEpcctCpRqDX9x).
  General access is SJSU with Viewer permission: sign in with an SJSU account.
  If you cannot access it, request read access from the folder owner.
- Extract `TinyStories-train.txt` and `TinyStories-valid.txt` directly into
  `task1_llm/data/`. Avoid an extra `TinyStories/` directory level; ignore any
  `__MACOSX` folder. The final paths must be:

  ```text
  task1_llm/data/TinyStories-train.txt
  task1_llm/data/TinyStories-valid.txt
  ```

  These are the original text files in
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main),
  not the V2-GPT4 files. Source sizes and SHA-256 values are recorded in
  [preprocessing_manifest.json](../../reproducibility/manifests/Yuyao_Ding/preprocessing_manifest.json).
  The smoke command runs notebook Section 1.1 automatically if preprocessing is missing.
- To reuse Mac preprocessing, copy the full `data_processed/` folder and its
  matching `configs/preprocessing.json` and
  `reproducibility/manifests/Yuyao_Ding/preprocessing_manifest.json` together.
  Then start at Section 1.2; raw text is unnecessary for this route.
- To evaluate or resume a Mac run, also copy its `checkpoints/<run_id>/` folder,
  including the checkpoint named by `best.json`. Keep the same repository-relative
  paths. Existing reports alone cannot restore model weights.

Transfer JSON files without changing their encoding or line endings: hashes
cover the file bytes. New JSON is written as UTF-8 with LF line endings.

## Run

`configs/preprocessing.json` controls sampling counts, seeds, story separator
and context length. Defaults are 100,000 training stories, 10,000 validation
stories and 256 characters per window. After a preprocessing change, rerun
Section 1.1 and keep `vocab_size` and `max_seq_len` in `configs/model.json` consistent
with its output. Model and training settings are in their respective JSON files.

After downloading the raw data, run the short check from the repository root.

Windows PowerShell:

```powershell
.\.venv\Scripts\python task1_llm/Yuyao_Ding/src/smoke_test.py
```

Mac or Linux:

```bash
.venv/bin/python task1_llm/Yuyao_Ding/src/smoke_test.py
```

The first run prepares the full configured subsets if the arrays or their manifest
are missing; later runs reuse them. This checks the model, trains for three optimizer
steps, reloads a checkpoint and generates a short sample using both decoders.
It uses the configured device and writes a separate `smoke_` run. Look for
`Training device: cuda`, `mps` or `cpu`, followed by `Smoke test passed:`.
Smoke results are pipeline checks, not formal model-quality results.

For full training, set `RUN_FORMAL_TRAINING = True` in Section 1.3.4, save the
notebook and restart the kernel before running Sections 1.2 onward (or run all
sections if preprocessing is needed). Full training defaults to 10 epochs.
`RESUME_CHECKPOINT` accepts a repository-relative completed-epoch checkpoint;
keep the original settings and planned epoch total. A resume starts a new run
directory. Set `SELECTED_RUN` in Section 1.3.5 to evaluate an older completed run.

New runs save a copy of the notebook's code in their manifest directory.
This is the saved source, so save edits before starting a run. Keep checkpoints,
logs and manifests together when moving a run between computers.

## Results and checkpoints

All paths below are relative to the repository root:

- `task1_llm/Yuyao_Ding/outputs/<run_id>/`: `history.csv`, `best.json`, `latest.json`,
  and, after completion, `metrics.json`. Section 1.3.5 adds plots and generated samples.
- `task1_llm/Yuyao_Ding/checkpoints/<run_id>/`: one `.pt` file per completed epoch.
- `reproducibility/raw_logs/Yuyao_Ding/<run_id>.jsonl`: the unedited training log.
- `reproducibility/manifests/Yuyao_Ding/<run_id>/manifest.json`: configurations,
  data fingerprints, hardware and checkpoint mapping. `source_snapshot.py` is saved
  for new runs.

The reported Windows run is `train_20260927T000510Z_b9df2210` (September 26 local time,
September 27 UTC). Its manifest is `complete`; epoch 10 is the best checkpoint.
The historical Mac run is `train_20260914T211231Z_98b2aa86`. Its logs and results remain
available, but its checkpoint files are absent from this copy.

The combined `Yuyao_Ding_Lab1_Resources.zip` includes this run's best/last checkpoint, vocabulary, split hashes, configs, source, results, original logs and manifests. Raw TinyStories text and encoded arrays are restored using the Drive and preprocessing instructions above. `SHA256SUMS.txt` at the project root verifies the package.

If preprocessing hashes fail after moving files, restore the matching artifacts
without changing JSON encoding or line endings. If you intentionally change the
preprocessing configuration, rerun Section 1.1 and update the model's vocabulary
size and context length before starting a new training run.

## Demo from the saved checkpoint

Run from the repository root. This loads the saved model and matching vocabulary,
then prints greedy and temperature-sampled text. It does not load training arrays,
start training, or replace the recorded report samples.

Windows PowerShell:

```powershell
.\.venv\Scripts\python task1_llm/Yuyao_Ding/src/demo.py --prompt "Once upon a time,"
```

Mac or Linux:

```bash
.venv/bin/python task1_llm/Yuyao_Ding/src/demo.py --prompt "Once upon a time,"
```

The default is the reported RTX 4090 checkpoint, 300 new characters, temperature
0.8 and seed 640. Device selection is CUDA, then MPS, then CPU. Use `--device cpu`
for a CPU demo, `--max-new-tokens 100` for a shorter sample, or `--run <run_id>`
for another restored run. Outputs on different devices need not be identical.
The script reuses the notebook's model and generation definitions.

For the viva, be ready to explain the causal mask, shifted input/target windows,
manual attention and LayerNorm, warm-up/cosine schedule, and why evaluation turns
off dropout. Explain that accuracy and n-gram diversity are character-level,
that the generalization gap uses the same checkpoint on both splits, and that
loss convergence does not guarantee a coherent story. The three exact failure
snippets are in `failure_analysis.md`. Teammate differences and comparison still
need the teammate's actual results.
