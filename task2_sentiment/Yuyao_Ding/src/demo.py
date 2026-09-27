"""Predict sentiment from a saved Task 2 model without loading data or training."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def load_model(root, session_id, name, requested_device="cpu"):
    member = root / "task2_sentiment/Yuyao_Ding"
    evaluation = json.loads(
        (member / "outputs" / session_id / name / "evaluation.json").read_text(encoding="utf-8")
    )
    checkpoint = root / evaluation["checkpoint"]
    with checkpoint.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    if checksum != evaluation["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != name or saved["run_id"] != evaluation["run_id"]:
        raise ValueError("Checkpoint and evaluation identify different models.")

    # Reuse the exact training definitions, without executing notebook cells.
    source = root / "reproducibility/manifests/Yuyao_Ding" / session_id / "source_notebook.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "parameters" not in cell["metadata"].get("tags", [])
    )
    if hashlib.sha256(code.encode("utf-8")).hexdigest() != saved["source_hashes"]["notebook_code"]:
        raise ValueError("The saved notebook does not match the checkpoint.")
    class_name = {
        "baseline": "MeanPoolClassifier",
        "textcnn": "TextCNNClassifier",
        "bilstm": "BiLSTMClassifier",
    }[name]
    names = {class_name, "preprocess_text", "select_device"}
    namespace = {
        "torch": torch, "nn": nn, "re": re,
        "pack_padded_sequence": pack_padded_sequence,
        "STOP_WORDS": set(saved["preprocessing"]["stop_words"]),
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if tree.body:
            exec(compile(tree, source.name, "exec"), namespace)
    if not names.issubset(namespace):
        raise ValueError("The saved notebook is missing a required definition.")
    device = namespace["select_device"](requested_device)
    model = namespace[class_name](**saved["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    return model, saved, namespace["preprocess_text"]


def predict(model, saved, preprocess, text):
    vocabulary = saved["vocabulary"]
    stoi = {token: index for index, token in enumerate(vocabulary["itos"])}
    tokens = preprocess(text).split()
    max_length = saved["preprocessing"]["max_length"]
    ids = [stoi.get(token, vocabulary["unk_idx"]) for token in tokens]
    ids = (ids or [vocabulary["unk_idx"]])[:max_length]
    ids += [vocabulary["pad_idx"]] * (max_length - len(ids))
    device = next(model.parameters()).device
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.tensor([ids], device=device))).item()
    threshold = saved["training_config"]["decision_threshold"]
    return {
        "model": saved["model_name"], "run_id": saved["run_id"],
        "epoch": saved["epoch"], "device": str(device),
        "sentiment": "positive" if probability >= threshold else "negative",
        "probability_positive": probability, "threshold": threshold,
        "processed_tokens": len(tokens), "truncated": len(tokens) > max_length,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="The food was good, but the service was not.")
    parser.add_argument("--model", choices=("baseline", "textcnn", "bilstm", "all"), default="bilstm")
    parser.add_argument("--run", help="Session ID; defaults to the latest complete formal run.")
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Enter a nonempty review.")
    root = Path(__file__).resolve().parents[3]
    latest = root / "task2_sentiment/Yuyao_Ding/outputs/latest.json"
    session_id = args.run or json.loads(latest.read_text(encoding="utf-8"))["session_id"]
    names = ("baseline", "textcnn", "bilstm") if args.model == "all" else (args.model,)
    for name in names:
        model, saved, preprocess = load_model(root, session_id, name, args.device)
        print(json.dumps(predict(model, saved, preprocess, args.text), indent=2))


if __name__ == "__main__":
    main()
