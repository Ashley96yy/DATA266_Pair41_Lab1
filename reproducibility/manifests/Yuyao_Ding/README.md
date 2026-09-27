# Run manifests

For each real run, record task/model/run ID, source commit, config and seed, dataset/split identification, actual package versions and environment file, exact CPU/GPU, start/end time, original raw-log location, checkpoint ID/location, and the metrics/report rows produced from that checkpoint. Record download/restore instructions for externally stored artifacts.

Task 1 has recorded runs. Start with the [setup and reproduction guide](../../../task1_llm/Yuyao_Ding/README.md).

- `train_20260914T211231Z_98b2aa86/manifest.json`: the historical Apple M3 run, retained for comparison. Its original checkpoints are preserved in this local checkout under `task1_llm/Yuyao_Ding/checkpoints/train_20260914T211231Z_98b2aa86/` (relative to the repository root).
- `train_20260927T000510Z_b9df2210/manifest.json`: the completed RTX 4090 run used by the current reports. Its saved source and best checkpoint are available locally.
- `smoke_*/manifest.json`: short pipeline checks, not formal results.
- `preprocessing_manifest.json`: the current preprocessing inputs, outputs and checksums.
- `model_implementation_checks.json`: the latest model checks.
- `setup_20260926T235405Z/`: Windows data verification and copies of the previous preprocessing/model-check records.

Keep completed run manifests and raw logs unchanged. When reporting a different run, use its own checkpoint and metrics rather than changing the historical evidence.

The Task 1 `outputs/train_20260927T000510Z_b9df2210/report_audit.json` records post-run
aggregate calculations and input checksums. It is derived report evidence; it does
not replace or modify the original run manifest or raw log.
