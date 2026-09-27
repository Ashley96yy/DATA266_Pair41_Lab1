"""Run the notebook's model checks, short training, and generation."""
import json
import os
from pathlib import Path


def main():
    notebook_path = Path(__file__).resolve().with_name("task1_llm.ipynb")
    repo_root = notebook_path.parents[3]
    os.chdir(repo_root)
    artifact_dir = notebook_path.parent.parent / "data_processed"
    required = ["train_ids.npy", "valid_ids.npy", "vocabulary.json"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    def run_cell(cell):
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path.name}:{cell['id']}", "exec"), namespace)

    manifest_path = repo_root / "reproducibility/manifests/Yuyao_Ding/preprocessing_manifest.json"
    if any(not (artifact_dir / name).is_file() for name in required) or not manifest_path.is_file():
        raw_dir = repo_root / "task1_llm/data"
        if any(not (raw_dir / name).is_file() for name in ("TinyStories-train.txt", "TinyStories-valid.txt")):
            raise FileNotFoundError(
                "Download TinyStories.zip and place both .txt files directly in task1_llm/data/. "
                "See task1_llm/Yuyao_Ding/README.md."
            )
        print("Preparing data with notebook Section 1.1; existing configurations are used.", flush=True)
        preprocessing_end = next(i for i, cell in enumerate(notebook["cells"])
                                 if cell["id"] == "gpt-code-12")
        for cell in notebook["cells"][:preprocessing_end]:
            if cell["cell_type"] == "code":
                run_cell(cell)

    cell_ids = [f"gpt-code-{i}" for i in (12, 14, 16, 18, 20, 22)]
    cell_ids += [f"train-code-{i}" for i in (24, 26, 28, 33)]
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell_id in cell_ids:
        run_cell(cells[cell_id])

    config = namespace["training_config"]
    run_dir = namespace["run_training"](config, smoke=True)
    namespace["generate_from_run"](
        run_dir, ["Once upon a time,"], max_new_tokens=20,
        device_name=config["device"], seed=config["seed"],
    )
    print("Smoke test passed:", run_dir.relative_to(repo_root))


if __name__ == "__main__":
    main()
