"""Translate one image with a saved generator; no training data or metric weights needed."""
import argparse
from pathlib import Path

import torch

from cyclegan import image_tensor, latest_checkpoint, load_generators, save_image, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=["A2B", "B2A"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")
    if args.output.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        parser.error("Output must be a JPG or PNG file.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = select_device(args.device)
    generators, state = load_generators(args.checkpoint or latest_checkpoint(), device)
    x = image_tensor(args.image, state["config"]["image_size"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        y = generators["G_" + args.direction](x)[0]
    if not torch.isfinite(y).all():
        raise FloatingPointError("Non-finite generator output.")
    save_image(y, args.output)
    print(f"{state['run_id']} | epoch {state['epoch']} | {args.direction} | {device}")
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
