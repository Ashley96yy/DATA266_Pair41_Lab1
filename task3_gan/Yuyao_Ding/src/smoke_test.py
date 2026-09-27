"""Run the notebook on tiny subsets, with an optional interrupted/resumed comparison."""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient
import torch

from cyclegan import project_root, read_checkpoint, sha256, write_json


def execute_smoke(device, stop=None, resume=None, evaluation=True):
    root = project_root()
    source = Path(__file__).with_name("task3_gan.ipynb").resolve()
    notebook = nbformat.read(source, as_version=4)
    parameters = [cell for cell in notebook.cells if cell.cell_type == "code"
                  and "parameters" in cell.metadata.get("tags", [])]
    if len(parameters) != 1:
        raise ValueError("Expected one notebook parameter cell.")
    settings = {"USE_COLAB_DRIVE": False, "RUN_FORMAL_TRAINING": False,
                "PROJECT_ROOT": str(root), "RESUME_CHECKPOINT": str(resume) if resume else None,
                "DEVICE": device, "RUN_EVALUATION": evaluation, "STOP_AFTER_EPOCH": stop}
    parameters[0].source = "\n".join(f"{name} = {value!r}" for name, value in settings.items())
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.cells.append(nbformat.v4.new_code_cell(
        'print("SMOKE_RESULT=" + json.dumps({"status": run_status, "checkpoint": str(best_checkpoint), '
        '"evaluation": str(evaluation_output) if RUN_EVALUATION and run_status["status"] == "complete" else None}))'))
    output = source.parent.parent / "outputs/smoke_checks"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output / f"smoke_{stamp}_{uuid4().hex[:6]}.ipynb"
    client = NotebookClient(notebook, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    print(f"Running small notebook check: device={device}, stop={stop}, resume={resume}", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)
        print("Executed notebook:", target, flush=True)
    for cell in notebook.cells:
        for item in cell.get("outputs", []):
            for line in item.get("text", "").splitlines():
                if line.startswith("SMOKE_RESULT="):
                    result = json.loads(line.removeprefix("SMOKE_RESULT="))
                    result["executed_notebook"] = str(target)
                    return result
    raise RuntimeError("Notebook did not report a completed run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--check-resume", action="store_true")
    args = parser.parse_args()
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    root = project_root()
    notebook_path = Path(__file__).with_name("task3_gan.ipynb")
    original_hash = sha256(notebook_path)
    protected = [root / "task3_gan/Yuyao_Ding" / name for name in
                 ("metrics_report.csv", "full_metrics_report.csv", "outputs/latest_formal.json")]
    previous_hashes = {str(p): sha256(p) if p.exists() else None for p in protected}
    complete = execute_smoke(args.device)
    assert complete["status"]["status"] == "complete"
    assert complete["status"]["nan_or_inf_events"] == 0
    evaluation = Path(complete["evaluation"])
    assert (evaluation / "smoke_submission_example.csv").is_file()
    assert not (evaluation / "submission.csv").exists()
    report = {"status": "passed", "complete_run": complete, "resume_check": None}
    if args.check_resume:
        interrupted = execute_smoke(args.device, stop=1, evaluation=False)
        last = Path(interrupted["checkpoint"]).with_name("last.pt")
        resumed = execute_smoke(args.device, resume=last, evaluation=False)
        reference = read_checkpoint(Path(complete["checkpoint"]).with_name("last.pt"))
        restored = read_checkpoint(Path(resumed["checkpoint"]).with_name("last.pt"))
        differences = {}
        for model, values in reference["models"].items():
            difference = max(float((tensor - restored["models"][model][key]).abs().max())
                             for key, tensor in values.items())
            differences[model] = difference
            if difference > 1e-6:
                raise AssertionError(f"Resume differs from uninterrupted training: {model} {difference}")
        report["resume_check"] = {"interrupted": interrupted, "resumed": resumed,
                                   "maximum_weight_differences": differences}
    assert sha256(notebook_path) == original_hash, "Source notebook changed during smoke check."
    assert previous_hashes == {str(p): sha256(p) if p.exists() else None for p in protected}
    destination = root / "task3_gan/Yuyao_Ding/outputs/smoke_checks" / f"{complete['status']['run_id']}_checks.json"
    write_json(destination, report)
    print("Smoke check passed:", destination, flush=True)


if __name__ == "__main__":
    main()
