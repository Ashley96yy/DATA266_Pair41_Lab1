"""Generate text from a saved checkpoint without loading the training data."""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="train_20260927T000510Z_b9df2210")
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if not args.prompt or args.max_new_tokens < 1 or args.temperature <= 0:
        parser.error("Use a nonempty prompt, positive length, and positive temperature.")

    notebook_path = Path(__file__).resolve().with_name("task1_llm.ipynb")
    member_dir = notebook_path.parent.parent
    repo_root = notebook_path.parents[3]
    metrics = json.loads((member_dir / "outputs" / args.run / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = repo_root / metrics["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    vocab_bytes = (member_dir / "data_processed" / "vocabulary.json").read_bytes()
    if hashlib.sha256(vocab_bytes).hexdigest() != state["data"]["vocabulary_sha256"]:
        raise ValueError("Vocabulary does not match the checkpoint.")
    vocabulary = json.loads(vocab_bytes)

    # Reuse the notebook definitions without running its data or training cells.
    names = {"TokenPositionEmbedding", "ManualLayerNorm", "FeedForward",
             "CausalSelfAttention", "TransformerBlock", "CharacterGPT",
             "select_device", "synchronize", "generate_text"}
    namespace = {"torch": torch, "nn": nn, "math": math, "time": time}
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if tree.body:
            exec(compile(tree, f"{notebook_path.name}:{cell['id']}", "exec"), namespace)
    if not names.issubset(namespace):
        raise ValueError("The notebook is missing a model or generation definition.")

    device = namespace["select_device"](args.device)
    model = namespace["CharacterGPT"](state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    print("Run:", metrics["run_id"])
    print("Checkpoint:", metrics["checkpoint"])
    print("Device:", device)
    for greedy in (True, False):
        sample = namespace["generate_text"](
            model, args.prompt, vocabulary, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, greedy=greedy, seed=args.seed,
        )
        print(f"\n[{sample['method']}] {sample['text']}")


if __name__ == "__main__":
    main()
