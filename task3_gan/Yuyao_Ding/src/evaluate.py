"""Evaluate a saved CycleGAN, export the course CSV, and prepare the blind audit."""
import argparse
import csv
import json
import importlib.metadata
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist, cosine
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms as T
from threadpoolctl import threadpool_limits

from cyclegan import (image_tensor, latest_checkpoint, load_generators, project_root,
                     save_image, select_device, sha256, synchronize, tensor_image,
                     translate_files, write_json)


class InceptionFeatures:
    """The same torchvision weights and preprocessing as the course notebook."""
    def __init__(self, device):
        self.device = device
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.to(device).eval().requires_grad_(False)
        self.transform = T.Compose([
            T.Resize(299), T.CenterCrop(299), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def __call__(self, paths, batch_size=16):
        if not paths:
            raise ValueError("Feature extraction needs at least one image.")
        features = []
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            batch = torch.stack(images).to(self.device)
            features.append(self.model(batch).float().cpu().numpy())
        result = np.concatenate(features)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite Inception features.")
        return result


def frechet_distance(real, generated):
    if min(len(real), len(generated)) < 2:
        raise ValueError("FID needs at least two samples in each set.")
    mu_r, mu_g = real.mean(axis=0), generated.mean(axis=0)
    cov_r, cov_g = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    # SciPy 1.18 removed disp=False. sqrtm now returns just the matrix.
    with threadpool_limits(limits=4):
        covmean = scipy.linalg.sqrtm(cov_r @ cov_g)
        if not np.isfinite(covmean).all():
            offset = np.eye(cov_r.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((cov_r + offset) @ (cov_g + offset))
    covmean = covmean.real if np.iscomplexobj(covmean) else covmean
    diff = mu_r - mu_g
    value = float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite FID.")
    return value


def course_mifid(real, generated):
    # This course uses sorted, index-matched cosine distances, not nearest neighbours.
    n = min(len(real), len(generated))
    value = float(np.mean([cosine(real[i], generated[i]) for i in range(n)]))
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite course MiFID.")
    return value


def kernel_distance(real, generated, subsets=50, subset_size=100, seed=41):
    """Unbiased polynomial-kernel MMD; negative finite estimates are possible."""
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(real), len(generated))
    if n < 2:
        raise ValueError("KID needs at least two samples per set.")
    estimates = []
    for _ in range(subsets):
        x = real[rng.choice(len(real), n, replace=False)].astype(np.float64)
        y = generated[rng.choice(len(generated), n, replace=False)].astype(np.float64)
        xx = (x @ x.T / x.shape[1] + 1) ** 3
        yy = (y @ y.T / y.shape[1] + 1) ** 3
        xy = (x @ y.T / x.shape[1] + 1) ** 3
        value = ((xx.sum() - np.trace(xx) + yy.sum() - np.trace(yy)) / (n * (n - 1))
                 - 2 * xy.mean())
        estimates.append(float(value))
    return {"KID": float(np.mean(estimates)), "KID_subset_std": float(np.std(estimates)),
            "KID_subset_size": n}


def distribution_metrics(real, generated, config):
    n = min(len(real), len(generated))
    real, generated = real[:n], generated[:n]
    k = config["nearest_k"]
    if n <= k:
        raise ValueError(f"Precision/recall/density/coverage need more than {k} samples.")
    rr, gg, rg = cdist(real, real), cdist(generated, generated), cdist(real, generated)
    # Index zero is the point itself, so index k is its kth neighbour.
    r_radius = np.partition(rr, k, axis=1)[:, k]
    g_radius = np.partition(gg, k, axis=1)[:, k]
    inside_real = rg < r_radius[:, None]
    inside_generated = rg < g_radius[None, :]
    result = {
        "FID": frechet_distance(real, generated), "MiFID": course_mifid(real, generated),
        "generative_precision": float(inside_real.any(axis=0).mean()),
        "generative_recall": float(inside_generated.any(axis=1).mean()),
        "density": float(inside_real.sum(axis=0).mean() / k),
        "coverage": float((rg.min(axis=1) < r_radius).mean()),
        "n_real": n, "n_generated": n,
    }
    result.update(kernel_distance(real, generated, config["kid_subsets"],
                                  config["kid_subset_size"], config["seed"]))
    return result


@torch.inference_mode()
def reconstruction_metrics(generators, paths, direction, device, size, lpips_model,
                           metric_device, output):
    forward = generators["G_" + direction]
    backward = generators["G_B2A" if direction == "A2B" else "G_A2B"]
    l1, perceptual, cycle_perceptual = [], [], []
    for index, path in enumerate(paths):
        x = image_tensor(path, size).unsqueeze(0).to(device)
        y = forward(x)
        cycle = backward(y)
        l1.append(float((cycle - x).abs().mean().cpu()) / 2)
        a, b, c = x.to(metric_device), y.to(metric_device), cycle.to(metric_device)
        perceptual.append(float(lpips_model(a, b).mean().cpu()))
        cycle_perceptual.append(float(lpips_model(a, c).mean().cpu()))
        if index < 6:
            panel = Image.new("RGB", (size * 3, size))
            for column, tensor in enumerate((x[0], y[0], cycle[0])):
                panel.paste(tensor_image(tensor), (size * column, 0))
            panel.save(output / f"{direction}_{Path(path).stem}_input_translation_cycle.png")
    return {"cycle_L1_0_to_1": float(np.mean(l1)),
            "LPIPS_input_translation_alex": float(np.mean(perceptual)),
            "LPIPS_input_cycle_alex": float(np.mean(cycle_perceptual))}


def prepare_audit(root, output, split, translated, size):
    audit = output / "audit"
    images = audit / "blinded"
    images.mkdir(parents=True)
    rows, key = [], []
    for index, sample in enumerate(split["audit_samples"], 1):
        case = f"case_{index:02d}"
        source = root / sample["path"]
        direction = sample["direction"]
        generated = translated[direction] / source.name
        if not generated.is_file():
            raise FileNotFoundError(f"Audit sample was not generated: {source.name}")
        save_image(image_tensor(source, size), images / f"{case}_input.jpg")
        # Preserve the direct exported model output; do not recompress it.
        (images / f"{case}_translation.jpg").write_bytes(generated.read_bytes())
        rows.append({"case_id": case, "target_style": "photo" if direction == "A2B" else "Monet",
                     "rater_name": "", "style": "", "content": "", "artifact_free": "", "comment": ""})
        key.append({"case_id": case, **sample})
    for number in (1, 2):
        with (audit / f"rater_{number}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "audit_key.json", key)
    (audit / "instructions.md").write_text(
        "# Independent blind review\n\n"
        "Two real people independently score the same fixed cases. Do not show them the run, "
        "checkpoint or audit_key.json, and do not discuss scores until both CSVs are complete. "
        "The left/source file is the input; translation is the model output. Target style is "
        "given in the CSV. This audit is blind to model identity, not to translation direction.\n\n"
        "Use integer scores 1-5: style (1=no target style, 3=partial, 5=convincing); "
        "content (1=major structure lost, 3=partly preserved, 5=structure preserved); "
        "artifact_free (1=severe defects, 3=visible defects, 5=no noticeable defects). "
        "Use 2 and 4 for intermediate cases. Higher is better for all three columns. "
        "Add the same rater_name to every row of your own CSV. Comments are optional.\n\n"
        "The cases were fixed from held-out data before training. Scores are human judgments; "
        "blank forms do not count as a completed audit.\n", encoding="utf-8")


def export_metrics(evaluation_dir, root=None, update_summary=False):
    """Rebuild the metric tables from saved evaluation and audit results."""
    root = Path(root).resolve() if root else project_root()
    evaluation_dir = Path(evaluation_dir).resolve()
    evaluation_path = evaluation_dir / "evaluation.json"
    result = json.loads(evaluation_path.read_text(encoding="utf-8"))
    fields = ["model_id", "direction_or_slice", "metric", "value", "unit", "split", "checkpoint_id", "evidence_path"]
    evidence = str(evaluation_path.relative_to(root))
    rows = []
    for protocol, directions in result["protocols"].items():
        for direction, metrics in directions.items():
            for name, value in metrics.items():
                rows.append(dict(zip(fields, [result["run_id"], direction, name, value, "raw",
                                              protocol, result["checkpoint_sha256"], evidence])))
    for name, value in result["resource_metrics"].items():
        if isinstance(value, (int, float)):
            rows.append(dict(zip(fields, [result["run_id"], "training", name, value, "raw", "train",
                                          result["checkpoint_sha256"], evidence])))
    training_metrics = (
        "G_A2B_adversarial", "G_B2A_adversarial", "D_A", "D_B", "G_total",
        "cycle_A", "cycle_B", "identity_A", "identity_B",
        "gradient_norm_G_A2B", "gradient_norm_G_B2A", "gradient_norm_D_A", "gradient_norm_D_B",
    )
    for epoch in result["training_history"]:
        for name in training_metrics:
            if name in epoch:
                # Online epoch averages are not measurements of the selected checkpoint.
                unit = "l2_norm" if name.startswith("gradient_norm_") else "loss"
                rows.append(dict(zip(fields, [result["run_id"], f"epoch_{epoch['epoch']:03d}",
                                              name, epoch[name], unit, "train", "", evidence])))
    audit = result.get("human_audit")
    if isinstance(audit, dict) and audit.get("status") == "complete":
        audit_path = evaluation_dir / audit["file"]
        scores = json.loads(audit_path.read_text(encoding="utf-8"))
        for direction, dimensions in {"overall": scores["metrics"], **scores["by_direction"]}.items():
            for dimension, metrics in dimensions.items():
                for metric, value in metrics.items():
                    rows.append(dict(zip(fields, [result["run_id"], direction,
                                                  f"human_{dimension}_{metric}", value,
                                                  "score_1_to_5" if metric == "mean" else "ratio",
                                                  "fixed_blind_audit", result["checkpoint_sha256"],
                                                  str(audit_path.relative_to(root))])))
    member = root / "task3_gan/Yuyao_Ding"
    latest = member / "outputs/latest_evaluation.json"
    if latest.exists():
        current = json.loads(latest.read_text(encoding="utf-8"))
        update_summary |= (root / current["path"].replace("\\", "/")).resolve() == evaluation_dir
    targets = [evaluation_dir / "full_metrics_report.csv"]
    if result["mode"] == "formal" and update_summary:
        targets += [member / "full_metrics_report.csv", member / "metrics_report.csv"]
    for target in targets:
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def aggregate_audit(evaluation_dir, rater1, rater2):
    evaluation_dir = Path(evaluation_dir).resolve()
    key = json.loads((evaluation_dir / "audit_key.json").read_text(encoding="utf-8"))
    expected = [row["case_id"] for row in key]
    scores, names = [], []
    for file in (rater1, rater2):
        with Path(file).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        mapping = {row["case_id"]: row for row in rows}
        if len(rows) != len(expected) or set(mapping) != set(expected):
            raise ValueError("Each rater must score every audit case exactly once.")
        rater_names = {row["rater_name"].strip() for row in rows}
        if len(rater_names) != 1 or not next(iter(rater_names)):
            raise ValueError("Fill one consistent rater_name in each CSV.")
        names.append(next(iter(rater_names)))
        values = np.array([[int(mapping[case][field]) for field in
                            ("style", "content", "artifact_free")] for case in expected])
        if not np.isin(values, [1, 2, 3, 4, 5]).all():
            raise ValueError("Human ratings must be integers from 1 to 5.")
        scores.append(values)
    if names[0].casefold() == names[1].casefold():
        raise ValueError("The audit requires two different human raters.")
    result = {"raters": names, "n_cases": len(expected), "metrics": {}, "by_direction": {},
              "rating_files": [{"file": str(Path(p).resolve()), "sha256": sha256(p)}
                               for p in (rater1, rater2)]}
    groups = {"overall": list(range(len(key))),
              **{direction: [i for i, row in enumerate(key) if row["direction"] == direction]
                 for direction in ("A2B", "B2A")}}
    for group, indices in groups.items():
        values = {}
        for column, name in enumerate(("style", "content", "artifact_free")):
            x, y = scores[0][indices, column], scores[1][indices, column]
            observed = float((x == y).mean())
            chance = sum(float((x == label).mean() * (y == label).mean()) for label in range(1, 6))
            values[name] = {
                "mean": float(np.concatenate([x, y]).mean()), "exact_agreement": observed,
                "cohen_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
            }
        if group == "overall":
            result["metrics"] = values
        else:
            result["by_direction"][group] = values
    result["note"] = "Kappa is undefined when both raters use one identical score for all cases; exact agreement is still reported."
    write_json(evaluation_dir / "audit_results.json", result)
    evaluation_path = evaluation_dir / "evaluation.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["human_audit"] = {"status": "complete", "file": "audit_results.json",
                                      "sha256": sha256(evaluation_dir / "audit_results.json")}
        write_json(evaluation_path, evaluation)
        export_metrics(evaluation_dir)
        status_path = evaluation_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["human_audit"] = "complete"
        write_json(status_path, status)
    return result


def evaluate_checkpoint(checkpoint, requested_device="auto", root=None):
    import lpips

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(root) if root else project_root()
    member = root / "task3_gan/Yuyao_Ding"
    device = select_device(requested_device)
    metric_device = torch.device("cpu") if device.type == "mps" else device
    torch.hub.set_dir(str(root / "task3_gan/data/metric_weights"))
    generators, state = load_generators(checkpoint, device)
    config, split = state["config"], state["split"]
    smoke = state["mode"] == "smoke"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = member / "outputs" / state["run_id"] / f"evaluation_{stamp}_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "mode": state["mode"]})
    try:
        features = InceptionFeatures(metric_device)
        perceptual = lpips.LPIPS(net="alex", verbose=False).to(metric_device).eval().requires_grad_(False)
        test = {domain: [root / path for path in split["test"][domain]] for domain in ("A", "B")}
        count = min(len(test["A"]), len(test["B"]), 8 if smoke else 300)
        test = {domain: sorted(paths)[:count] for domain, paths in test.items()}
        course_protocol = "course_kaggle_smoke" if smoke else "course_kaggle"
        protocols = {"heldout_test": test, course_protocol: {
            domain: sorted((root / f"task3_gan/data/{folder}").glob("*.jpg"))[:8 if smoke else 300]
            for domain, folder in (("A", "monet_jpg"), ("B", "photo_jpg"))}}
        inventory_path = root / "reproducibility/manifests/Yuyao_Ding/task3_data_inventory.json"
        if sha256(inventory_path) != split["data_inventory_sha256"]:
            raise ValueError("The raw-data inventory differs from the training run.")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        expected_hashes = {row["path"]: row["sha256"] for row in inventory["images"]}
        for inputs in protocols.values():
            for paths in inputs.values():
                for path in paths:
                    if sha256(path) != expected_hashes[path.relative_to(root).as_posix()]:
                        raise ValueError(f"Evaluation image changed: {path.name}")
        result = {
            "run_id": state["run_id"], "mode": state["mode"], "epoch": state["epoch"],
            "checkpoint": str(Path(checkpoint).resolve().relative_to(root)),
            "checkpoint_sha256": sha256(checkpoint), "config": config,
            "split_sha256": state["split_sha256"], "protocols": {},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "device": str(device), "metric_device": str(metric_device),
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("torch", "torchvision", "numpy", "scipy", "lpips", "Pillow")}},
            "source_sha256": {"evaluate.py": sha256(__file__),
                              "official_notebook": sha256(root / "task3_gan/Part3_Evaluation_Script.ipynb")},
            "metric_weights_sha256": {p.name: sha256(p) for p in
                                      (root / "task3_gan/data/metric_weights/checkpoints").glob("*.pth")},
            "resource_metrics": state["resources"], "human_audit": "pending",
            "training_history": state["history"],
            "training_curves": str((member / "outputs" / state["run_id"] / "training_curves.png").relative_to(root)),
            "kaggle_public_score": None, "kaggle_private_score": None,
            "metric_notes": {
                "features": "torchvision Inception-v3 IMAGENET1K_V1; fc replaced by identity; course preprocessing",
                "FID": "Course covariance FID; sqrtm without the removed SciPy disp argument",
                "MiFID": "Course mean cosine distance after sorting and matching by index; not nearest-neighbour MiFID",
                "KID": "Unbiased degree-3 polynomial MMD; raw scale, not multiplied by 1000; subset std is not a confidence interval",
                "cycle_L1": "Direct FP32 cycle reconstruction; scaled to pixel range [0,1]",
                "LPIPS": "AlexNet v0.1, source vs translation and source vs cycle; no paired target exists",
                "content_cosine": "Inception cosine between each source and its exported JPEG translation",
                "heldout_test": "Balanced fixed test subset; no training or checkpoint selection; small Monet test set makes distribution metrics noisy",
                "course_kaggle": "First 300 sorted real and source filenames per domain, including training images as required by the supplied reference; not a held-out generalization estimate",
            },
        }
        evaluation_start = time.perf_counter()
        for protocol, inputs in protocols.items():
            destination = output / protocol
            destination.mkdir()
            real_features = {domain: features(paths, config["evaluation_batch_size"])
                             for domain, paths in inputs.items()}
            translated = {}
            result["protocols"][protocol] = {}
            for direction, source_domain, target_domain in (("A2B", "A", "B"), ("B2A", "B", "A")):
                print(f"Evaluating {protocol} {direction} ({len(inputs[source_domain])} images)", flush=True)
                translated[direction] = destination / f"pred_{direction}"
                synchronize(device)
                started = time.perf_counter()
                generated = translate_files(generators["G_" + direction], inputs[source_domain],
                                            translated[direction], device, config["image_size"],
                                            config["batch_size"])
                synchronize(device)
                elapsed = time.perf_counter() - started
                generated_features = features(generated, config["evaluation_batch_size"])
                metrics = distribution_metrics(real_features[target_domain], generated_features, config)
                a, b = real_features[source_domain], generated_features
                metrics["content_preservation_cosine"] = float(np.mean(
                    np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)).clip(1e-12)))
                metrics["inference_images_per_second_including_io"] = len(generated) / elapsed
                if protocol == "heldout_test":
                    metrics.update(reconstruction_metrics(generators, inputs[source_domain], direction,
                                                           device, config["image_size"], perceptual,
                                                           metric_device, destination))
                result["protocols"][protocol][direction] = metrics
                write_json(destination / f"{direction}_images.json", [
                    {"source": str(p.relative_to(root)), "source_sha256": sha256(p),
                     "generated": str(g.relative_to(root)), "generated_sha256": sha256(g)}
                    for p, g in zip(inputs[source_domain], generated)])
            if protocol == "heldout_test":
                if not smoke:
                    prepare_audit(root, output, split, translated, config["image_size"])
                elif split["audit_samples"]:
                    prepare_audit(root, output, split, translated, config["image_size"])
        scores = result["protocols"][course_protocol]
        row = {"ID": 1, "FID": float(np.mean([v["FID"] for v in scores.values()])),
               "MiFID": float(np.mean([v["MiFID"] for v in scores.values()]))}
        csv_path = output / ("smoke_submission_example.csv" if smoke else "submission.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ID", "FID", "MiFID"])
            writer.writeheader()
            writer.writerow(row)
        result["evaluation_seconds"] = time.perf_counter() - evaluation_start
        write_json(output / "evaluation.json", result)
        export_metrics(output, root, update_summary=not smoke)
        write_json(output / "status.json", {"status": "complete", "mode": state["mode"],
                                            "human_audit": "pending", "kaggle_submission": "not submitted"})
        if not smoke:
            write_json(member / "outputs/latest_evaluation.json", {"path": str(output.relative_to(root)),
                                                                   "checkpoint_sha256": result["checkpoint_sha256"]})
        print("Evaluation saved:", output, flush=True)
        return output, result

    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "mode": state["mode"],
                                            "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--export-only", action="store_true", help="Rebuild CSVs from saved results without inference.")
    parser.add_argument("--rater1", type=Path)
    parser.add_argument("--rater2", type=Path)
    args = parser.parse_args()
    if args.export_only:
        if not args.evaluation_dir or args.rater1 or args.rater2 or args.checkpoint:
            parser.error("Use --export-only with --evaluation-dir, without checkpoint or rater arguments.")
        print(f"Exported {export_metrics(args.evaluation_dir)} metric rows.")
    elif args.rater1 or args.rater2 or args.evaluation_dir:
        if not all((args.rater1, args.rater2, args.evaluation_dir)):
            parser.error("Provide --evaluation-dir, --rater1 and --rater2 together.")
        print(json.dumps(aggregate_audit(args.evaluation_dir, args.rater1, args.rater2), indent=2))
    else:
        evaluate_checkpoint(args.checkpoint or latest_checkpoint(), args.device)


if __name__ == "__main__":
    main()
