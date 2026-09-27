# Task 2 — Yuyao Ding

This combined archive uses an environment inside the extracted project. Start with the [project setup](../../README.md). In older Windows examples below, replace `..\.venv\Scripts\python` with `.\.venv\Scripts\python`. The original computer kept its environment one directory above the project.


Training and evaluation stay in [task2_sentiment.ipynb](src/task2_sentiment.ipynb).
The notebook runs independently and contains the three model definitions.
Two small optional entry points serve separate purposes:

- [demo.py](src/demo.py) loads a saved checkpoint and predicts a review without training or raw data.
- [smoke_test.py](src/smoke_test.py) runs the notebook on its small subset without editing the saved notebook.

Configuration, data, checkpoints and results remain external files. Development
scripts from the earlier implementation are archived outside the project.

## Download processed data and checkpoints

These artifacts are stored in Google Drive and are not included in a Git clone.
Restore the folder contents to the following paths relative to the repository root:

| Artifact | Google Drive | Restore destination | Details |
| --- | --- | --- | --- |
| Processed data | [Download folder](https://drive.google.com/drive/folders/1jzNHg-lIQ1ok9_8g7d_9rF2HsN9JOaO_?usp=sharing) | `task2_sentiment/Yuyao_Ding/data_processed/` | [Layout and required files](data_processed/README.md) |
| Checkpoints | [Download folder](https://drive.google.com/drive/folders/18AzMJ8NbMmE7l38yGKZziBppHmq4LlIZ?usp=sharing) | `task2_sentiment/Yuyao_Ding/checkpoints/` | [Run IDs and model selection](checkpoints/README.md) |

Preserve nested run directories and file bytes; avoid an extra directory level
when extracting a downloaded folder. Use an account with read access. These links
are for Yuyao's artifacts; raw datasets are listed separately in the data section.

## Completed experiment

The main session is `task2_formal_20260927T020054Z_ba87cd57`. All three models completed
five epochs and evaluation on 38,000 test reviews. Baseline / TextCNN / BiLSTM test
accuracy is 93.24% / 93.88% / 94.75%; the selected epochs are 5 / 3 / 3.
[Results](results.md), [case analysis](failure_analysis.md), and
[the report contribution](../../report/Task2_Yuyao_Ding.md) use this run.

The saved notebook includes the completed formal outputs and currently has
RUN_FORMAL_TRAINING = True. Opening it only displays those results; Run All starts
a new experiment. Use the demo below to predict without retraining.

## Predict from a checkpoint

On this Windows computer, run from the project root:

~~~powershell
..\.venv\Scripts\python task2_sentiment/Yuyao_Ding/src/demo.py --text "The food was good, but the service was not."
~~~

The default is the latest completed formal BiLSTM on CPU. Add `--model all` to compare
all three, or select `--model baseline` / `--model textcnn`. `--run SESSION_ID`
selects another saved session, and `--device auto` chooses CUDA, then MPS, then CPU.
CPU is sufficient for an individual review; use `--device cpu` if a device backend
does not support an operation. The output includes the run, epoch, probability,
decision threshold and whether the processed input was truncated.
Probabilities can differ slightly across devices and batch sizes. The CPU/CUDA
demo checks matched the saved labels on all 64 selected reviews per model; the
largest observed probability difference was about 0.00030. Full test metrics above
remain those from the original CUDA evaluation.

For an environment created inside the project, use `.\.venv\Scripts\python` on
Windows or `.venv/bin/python` on Mac/Linux with the same script and arguments.
The command works from other directories if the script path is absolute.

The demo checks the checkpoint and frozen notebook source hashes, then loads only
the saved model/preprocessing definitions. It uses the checkpoint vocabulary and
stopword list. Keep the project's relative layout: `src/demo.py`,
`outputs/latest.json`, `outputs/SESSION/MODEL/evaluation.json`,
`checkpoints/SESSION/MODEL/best.pt`, and
`reproducibility/manifests/Yuyao_Ding/SESSION/source_notebook.ipynb`.
Raw Yelp files, processed arrays and a running Jupyter kernel are not needed.

## Run on the current Windows computer

Open PowerShell in the project root (the folder containing task1_llm and
task2_sentiment). The existing environment is one directory above this folder:

~~~powershell
..\.venv\Scripts\python -m jupyterlab task2_sentiment/Yuyao_Ding/src/task2_sentiment.ipynb
~~~

The environment is already installed. Select its Python kernel. These steps are
for a new training run; they are not needed to inspect or demonstrate the saved run.

1. Put the Yelp data in the directory below.
2. Leave USE_COLAB_DRIVE = False and PROJECT_ROOT = None on this computer.
3. Run All with RUN_FORMAL_TRAINING = False for the small check.
4. For the actual experiment, set RUN_FORMAL_TRAINING = True, restart the
   kernel, and Run All. Save the executed notebook after completion.

The default smoke run uses a stratified subset, three training batches per
model and 50 bootstrap resamples. It does not fill the formal metrics report.
The final notebook section predicts an example review with the best models.

For a one-command smoke check on the current computer:

~~~powershell
..\.venv\Scripts\python task2_sentiment/Yuyao_Ding/src/smoke_test.py
~~~

On another Windows setup use `.\.venv\Scripts\python`; on Mac/Linux use
`.venv/bin/python`. The original Yelp files are required. The wrapper changes the
launch controls only in memory, executes the notebook's smoke path and saves a
separate executed copy under `outputs/smoke_checks`. It leaves the main notebook,
formal metrics and `outputs/latest.json` unchanged. Model logs and checkpoints use
new smoke session directories.

## Data from Drive

Download yelp_polarity.zip from the Task 2 data directory in the
[team Drive folder](https://drive.google.com/drive/folders/1Enmnu7482fsa0-9p_2rmzsyEImcwwMSg).
Sign in with an account that has read access, then place the ZIP here:

~~~text
task2_sentiment/data/yelp_polarity.zip
~~~

The notebook extracts just the expected parquet files, including ZIPs with an
enclosing folder. Alternatively, put these original files directly in data:

~~~text
task2_sentiment/data/train-00000-of-00001.parquet
task2_sentiment/data/test-00000-of-00001.parquet
~~~

Both files must have text and label columns, labels 0/1, and the original
560,000 / 38,000 row counts. The test set is retained in full for formal
training. Local execution reads the downloaded files; it does not sign into
Drive automatically.

## Setup on another computer

Use Python 3.12 and create the environment from the project root. Do not copy
an environment between operating systems. If the Task 1 environment already
exists, reuse it and install the Task 2 requirements.

Windows with NVIDIA GPU:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r task2_sentiment/Yuyao_Ding/requirements.txt
.\.venv\Scripts\python -m jupyterlab task2_sentiment/Yuyao_Ding/src/task2_sentiment.ipynb
~~~

Apple Silicon Mac:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r task2_sentiment/Yuyao_Ding/requirements.txt
python -m jupyterlab task2_sentiment/Yuyao_Ding/src/task2_sentiment.ipynb
~~~

Linux uses the same environment commands as Mac. For an NVIDIA GPU, install
torch==2.8.0 from the cu128 index before installing requirements. For Windows
or Linux CPU only, use the cpu index instead. See the
[official PyTorch installation choices](https://pytorch.org/get-started/previous-versions/#v280).

In configs/training.json, device auto selects CUDA, then MPS, then CPU. The
loaders use zero worker processes. Use device cpu if a Mac's MPS backend
cannot execute an operation. Cross-device floating-point results can differ.

On Colab, keep the full project on Drive, mount it, and install requirements:

~~~python
from google.colab import drive
drive.mount('/content/drive')
%pip install -r '/content/drive/MyDrive/DATA266_Pair41_Lab1/task2_sentiment/Yuyao_Ding/requirements.txt'
~~~

Restart the runtime if packages changed. In the notebook enable
USE_COLAB_DRIVE and set PROJECT_ROOT if the Drive folder has a different name.
With the project root on Drive, weights and results are saved there.

## Settings and results

The four JSON files in configs define preprocessing, model, training and
evaluation settings. Default embeddings are 128-dimensional and trained from
scratch; sequence length is 192 and vocabulary size is capped at 50,000.

| Model | Main difference | Dropout |
|---|---|---:|
| Baseline | Masked mean pooling | 0.30 |
| TextCNN | 128 filters each for widths 3/4/5; masked max pooling | 0.50 |
| BiLSTM | 128 hidden units per direction; packed sequences | 0.40 |

All models use Adam at 0.001, batch size 256, at most five epochs, and
validation-loss early stopping with patience two. BiLSTM clips gradient norm
at 1. Test evaluation uses the best validation checkpoint and a fixed 0.5 threshold.

Every run creates a new session ID. Under Yuyao_Ding:

- data_processed/SESSION contains processed splits, vocabulary and encoded inputs.
- checkpoints/SESSION/MODEL contains best.pt, numbered best checkpoints and last.pt.
- outputs/SESSION contains plots, predictions, metrics, slices and error-review CSVs.

Original logs and manifests are under reproducibility/raw_logs/Yuyao_Ding and
reproducibility/manifests/Yuyao_Ding at the project root. Manifests record data
hashes, the notebook source, configuration, hardware and installed packages.
Completed logs remain unchanged. A formal comparison updates metrics_report.csv
and outputs/latest.json; smoke results remain separate.

The notebook exports all required classification/calibration metrics, 95%
bootstrap intervals for accuracy/macro-F1/MCC, paired McNemar tests and slice
metrics. PR-AUC is trapezoidal area; average precision is separate. ECE uses
10 confidence bins and formal bootstrap uses 1,000 resamples. McNemar reports
exact two-sided p-values and Holm adjustment for the two comparisons.

CPU memory is process-lifetime peak RSS. CUDA memory is the allocator peak in
that training segment; MPS memory is sampled allocation, not a true peak.

## Resume and written analysis

To resume, set RESUME_CHECKPOINTS in the first code cell, for example:

~~~python
RESUME_CHECKPOINTS = {
    'baseline': 'task2_sentiment/Yuyao_Ding/checkpoints/PREVIOUS_SESSION/baseline/last.pt',
    'textcnn': 'task2_sentiment/Yuyao_Ding/checkpoints/PREVIOUS_SESSION/textcnn/last.pt',
}
~~~

Include only existing checkpoints. Keep the same data, mode, code and settings,
restart the kernel and Run All. Completed models reuse their saved training;
interrupted models continue after the last complete epoch with Adam and
PyTorch RNG state restored. Unfinished work inside an epoch is repeated.

For the completed main session, all three error_review.csv files now contain
case-specific error_type, observation and testable_fix entries. There are 20 distinct
errors per model across four categories, with no shortages. Results and analysis
are written in results.md and failure_analysis.md. A new experiment produces its
own candidates and will need a new analysis. The other member's final model choices
and results still need to be checked for team-wide distinctness and comparison.

Git excludes raw data, processed arrays and weights. Back up the matching
project, configuration, checkpoints, outputs and reproducibility records on
Drive before moving to another machine or relying on a temporary runtime.

For command-line execution of the notebook, first confirm the desired
RUN_FORMAL_TRAINING setting. With False this is a smoke check:

~~~bash
python -m jupyter nbconvert --to notebook --execute task2_sentiment/Yuyao_Ding/src/task2_sentiment.ipynb --output task2_executed.ipynb --output-dir task2_sentiment/Yuyao_Ding/outputs --ExecutePreprocessor.timeout=-1
~~~

## Validation status

The standalone notebook passed all 45 code cells in actual Jupyter kernels
on Windows CPU and RTX 4090 with synthetic fixtures. Eight regression checks cover
metrics, padding, data overlap, paired IDs and checkpoint recovery. The historical
[implementation checks](../../reproducibility/manifests/Yuyao_Ding/task2_implementation_checks.json)
predate the formal run and retain their original status snapshot.

Formal Yelp training and evaluation then completed on RTX 4090. Current demo,
smoke-wrapper and artifact checks are in the
[closeout record](../../reproducibility/manifests/Yuyao_Ding/task2_closeout_checks.json).
Mac/Linux/Colab have not been physically tested here. Recreate the virtual environment
on each operating system; do not copy a Windows .venv to Mac or Linux.

For reproduction, keep the full member folder plus its matching raw logs and manifests,
and the report contribution. Download raw data from the documented Drive location
for smoke checks or retraining. The combined `Yuyao_Ding_Lab1_Resources.zip`
contains the saved models and reproduction materials; share it using the team's
chosen transfer method.
