# Run manifests - Pratiksha Kaushik

One folder per final run. The raw logs for the same runs are in `../../raw_logs/Pratiksha_Kaushik/`.

| Task | Run | Hardware | Manifest | Raw log | Checkpoints |
|---|---|---|---|---|---|
| Task 1 - char-level GPT | `20261005-223946_L8H8C512T512` (seed 1337, 10 epochs) | NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, Python 3.13.15 | `task1_20261005-223946_L8H8C512T512/` (manifest json + md, environment, requirements lock) | `task1_pratiksha_kaushik_20261005-223946_L8H8C512T512_runlog_from_notebook_output.txt` (full log, 648 lines) | [Google Drive](https://drive.google.com/drive/folders/18IwhldDiOowviFXWqD22ZC2dRBwvGPEF?usp=drive_link) |
| Task 2 - Yelp sentiment (3 models) | final all-model run, 2026-10-05 01:19 (seed 9002, 5 epochs each) | NVIDIA Tesla T4 (Colab), PyTorch 2.11.0+cu130 | `task2_20261005_all_models/manifest.json` | `task2_pratiksha_kaushik_20261005_all_models_runlog.txt` (two runs; the second block is the final one) | [Google Drive](https://drive.google.com/drive/folders/1sBI2vdiaSnv6S1KcWrIeYHQg5oVAT_k4?usp=drive_link) |
| Task 3 - CycleGAN v3 | v3 final run, 2026-10-01 18:55 to 2026-10-02 01:05 (seed 9002, 80 epochs) | NVIDIA GeForce RTX 4090, PyTorch 2.1.2, Python 3.10.13 | `task3_20261001_cyclegan_v3/manifest.json` | `task3_pratiksha_kaushik_20261001_cyclegan_v3_runlog.txt` | [Google Drive](https://drive.google.com/drive/folders/1Ce8xc2Tt_9E9ZXQC9ou5aPAu1iLHpdq2?usp=drive_link) |

## Notes

- **Task 1:** the log file written during training (`..._runlog.txt`) stops after 14 setup lines. The full log was printed in the notebook, so `..._runlog_from_notebook_output.txt` contains those lines exactly as printed.
- **Task 2 and Task 3:** each manifest lists the hardware, versions, seed, data split, checkpoint info and the SHA-256 of every output file in that task's folder.
- The same files are kept in each task folder (`task1_llm/Pratiksha_Kaushik/reproducibility/`, and `RUN_LOG.txt` + `reproducibility_manifest.json` in the Task 2 and Task 3 folders).
