"""Run the notebook's small check without editing the saved training notebook."""
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient


def main():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    source = Path(__file__).resolve().with_name("task2_sentiment.ipynb")
    root = source.parents[3]
    notebook = nbformat.read(source, as_version=4)
    parameters = [cell for cell in notebook.cells
                  if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])]
    if len(parameters) != 1:
        raise ValueError("Expected one notebook parameter cell.")
    parameters[0].source = (
        "USE_COLAB_DRIVE = False\nRUN_FORMAL_TRAINING = False\n"
        f"PROJECT_ROOT = {str(root)!r}\nRESUME_CHECKPOINTS = {{}}\n"
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    print("Checking all three models on the notebook's small data subset...", flush=True)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    output_dir = source.parent.parent / "outputs/smoke_checks"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_dir / f"smoke_{stamp}_{uuid4().hex[:8]}.ipynb"
    nbformat.write(notebook, target)
    print("Smoke check passed:", target)


if __name__ == "__main__":
    main()
