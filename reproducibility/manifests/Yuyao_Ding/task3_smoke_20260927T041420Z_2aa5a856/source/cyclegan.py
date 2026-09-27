"""CycleGAN models and the image/checkpoint helpers shared by the notebook and demo."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, width=64, blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, width, 7),
                  nn.InstanceNorm2d(width), nn.ReLU(inplace=True)]
        for _ in range(2):
            layers += [nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(width * 2), nn.ReLU(inplace=True)]
            width *= 2
        layers += [ResidualBlock(width) for _ in range(blocks)]
        for _ in range(2):
            # Resize followed by convolution avoids uneven transpose-convolution overlap.
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.ReflectionPad2d(1), nn.Conv2d(width, width // 2, 3),
                       nn.InstanceNorm2d(width // 2), nn.ReLU(inplace=True)]
            width //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(width, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """70 x 70 PatchGAN; D_A judges Monet, and D_B judges photographs."""
    def __init__(self, width=64):
        super().__init__()
        layers = [nn.Conv2d(3, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
            previous = width if multiplier == 2 else width * (multiplier // 2)
            layers += [nn.Conv2d(previous, width * multiplier, 4, stride, 1),
                       nn.InstanceNorm2d(width * multiplier), nn.LeakyReLU(0.2, True)]
        layers.append(nn.Conv2d(width * 8, 1, 4, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def initialize(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_models(config, device):
    models = {
        "G_A2B": Generator(config["width"], config["residual_blocks"]),
        "G_B2A": Generator(config["width"], config["residual_blocks"]),
        "D_A": Discriminator(config["discriminator_width"]),
        "D_B": Discriminator(config["discriminator_width"]),
    }
    for model in models.values():
        model.apply(initialize)
        model.to(device)
    return models


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install the CUDA torch wheel.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu.")
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError("Device must be auto, cuda, mps or cpu.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def project_root():
    return Path(__file__).resolve().parents[3]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {"file": path.name, "sha256": sha256(path),
                                         "epoch": state["epoch"], "run_id": state["run_id"]})


def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if record["file"] != path.name or sha256(path) != record["sha256"]:
        raise ValueError("Checkpoint checksum does not match its JSON record.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["model_source_sha256"] != sha256(__file__):
        raise ValueError("The model source differs from the checkpoint. Use the saved source snapshot.")
    return state


def load_generators(path, device):
    state = read_checkpoint(path)
    config = state["config"]
    models = {}
    for name in ("G_A2B", "G_B2A"):
        model = Generator(config["width"], config["residual_blocks"])
        model.load_state_dict(state["models"][name], strict=True)
        models[name] = model.to(device).eval()
    return models, state


def latest_checkpoint(root=None):
    root = Path(root) if root else project_root()
    member = root / "task3_gan/Yuyao_Ding"
    record = json.loads((member / "outputs/latest_formal.json").read_text(encoding="utf-8"))
    return root / record["checkpoint"]


def image_tensor(path, size=256):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        return T.functional.to_tensor(image) * 2 - 1


def tensor_image(tensor):
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
    array = array.round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_image(tensor).save(path, quality=95, subsampling=0)


@torch.inference_mode()
def translate_files(model, paths, output, device, size=256, batch_size=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [Path(path).name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Input basenames must be unique within each domain.")
    written = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        inputs = torch.stack([image_tensor(p, size) for p in batch_paths]).to(device)
        outputs = model(inputs)
        if not torch.isfinite(outputs).all():
            raise FloatingPointError("Non-finite generator output.")
        for path, tensor in zip(batch_paths, outputs):
            target = output / Path(path).name
            save_image(tensor, target)
            written.append(target)
    return written
