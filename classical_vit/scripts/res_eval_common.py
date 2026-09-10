import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilitys.builder import build_model
from utilitys.data_process import build_dataloader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

C_VALUES = (2, 3, 4, 5, 6)
SNR_THRESHOLD = 12.0


def parse_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_config(cfg):
    cfg = dict(cfg)
    for key in ("size_image", "size_windows"):
        if isinstance(cfg.get(key), list):
            cfg[key] = tuple(cfg[key])

    if "dim_head_before_qkv_proj" not in cfg and "dim_head" in cfg:
        cfg["dim_head_before_qkv_proj"] = cfg["dim_head"]
    if "dim_head_before_qkv_proj" not in cfg and "dim_emb" in cfg:
        cfg["dim_head_before_qkv_proj"] = int(cfg["dim_emb"]) // int(cfg["num_heads"])

    cfg["qkv_projection"] = str(cfg.get("qkv_projection", "linear")).lower()
    return cfg


def read_run_config(run_dir):
    run_json = run_dir / "run.json"
    local_resolved_config = run_dir / "resolved_config.json"
    parent_resolved_config = run_dir.parent / "resolved_config.json"

    if run_json.exists():
        with open(run_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "config" in payload:
            return normalize_config(payload["config"])

    if local_resolved_config.exists():
        with open(local_resolved_config, "r", encoding="utf-8") as f:
            return normalize_config(json.load(f))

    if parent_resolved_config.exists():
        with open(parent_resolved_config, "r", encoding="utf-8") as f:
            return normalize_config(json.load(f))

    raise FileNotFoundError(f"No run.json or resolved_config.json found for {run_dir}")


def config_group_name(cfg):
    heads = int(cfg.get("num_heads", 4))
    dim_before = int(cfg["dim_head_before_qkv_proj"])
    emb = dim_before * heads
    projection = cfg.get("qkv_projection", "linear")

    if projection == "linear":
        dim_after = int(cfg.get("dim_head_after_qkv_proj", dim_before))
        return f"linear_emb{emb}_inner{dim_after * heads}"

    hidden = int(cfg["qkv_mlp_hidden_dim"])
    return f"mlp_emb{emb}_inner{hidden * heads}"


def iter_run_dirs_for_c(result_roots, c_value):
    run_dirs = []
    for root in result_roots:
        if not root.exists():
            print(f"[WARN] missing result root: {root}")
            continue

        for resolved_config in sorted(root.rglob("resolved_config.json")):
            family_dir = resolved_config.parent
            cfg = read_run_config(family_dir)
            if int(cfg["dim_head_before_qkv_proj"]) != int(c_value):
                continue
            for run_dir in sorted(d for d in family_dir.iterdir() if d.is_dir() and "seed" in d.name):
                run_dirs.append(run_dir)

    return run_dirs


def load_checkpoint_state(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_checkpoint_payload(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)


def load_model(run_dir, device):
    ckpt_path = run_dir / "best_model.pth"
    if not ckpt_path.exists():
        return None, None, None

    cfg = read_run_config(run_dir)
    model = build_model(cfg, device)
    checkpoint = load_checkpoint_payload(ckpt_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best = checkpoint.get("best", {})
        threshold = best.get("val_threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
    model.eval()
    return model, cfg, threshold


def build_test_transform(cfg):
    size_image = cfg["size_image"]
    short_side = min(size_image) if isinstance(size_image, (list, tuple)) else size_image
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize(short_side),
            transforms.CenterCrop(size_image),
            transforms.ToTensor(),
            transforms.Lambda(per_image_zscore),
        ]
    )


def per_image_zscore(img_tensor, eps=1e-6):
    mean = img_tensor.mean()
    std = img_tensor.std()
    return (img_tensor - mean) / (std + eps)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=2, threshold=0.5):
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        prob_frb = torch.softmax(outputs, dim=1)[:, 0]
        preds = torch.where(
            prob_frb >= threshold,
            torch.zeros_like(labels),
            torch.ones_like(labels),
        )

        for target, pred in zip(labels, preds):
            conf[target, pred] += 1

    tp = conf[0, 0].item()
    fn = conf[0].sum().item() - tp
    fp = conf[:, 0].sum().item() - tp
    tn = conf.sum().item() - (tp + fp + fn)

    total = conf.sum().item()
    acc = (tp + tn) / total if total else 0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f2 = 5 * precision * recall / (4 * precision + recall + 1e-8)

    return {
        "acc": round(acc * 100, 2),
        "recall": round(recall * 100, 2),
        "precision": round(precision * 100, 2),
        "f2": round(f2 * 100, 2),
    }


def evaluate_full(run_dir, test_dataset, device):
    model, cfg, threshold = load_model(run_dir, device)
    if model is None:
        return None, None

    test_dataset.transform = build_test_transform(cfg)
    loader = build_dataloader(
        dataset=test_dataset,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
    )
    return evaluate(model, loader, device, int(cfg.get("num_classes", 2)), threshold), cfg


def extract_snr(path):
    stem = Path(path).stem
    try:
        return float(stem.split("_")[-2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot extract SNR from file name: {path}") from exc


class SNRDataset(Dataset):
    def __init__(self, base_dataset, threshold, mode, transform=None):
        self.samples = []
        self.transform = transform

        for path, label in base_dataset.samples:
            snr = extract_snr(path)
            if mode == "low" and snr < threshold:
                self.samples.append((path, label))
            elif mode == "high" and snr >= threshold:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_snr(run_dir, base_dataset, device, snr_threshold):
    model, cfg, threshold = load_model(run_dir, device)
    if model is None:
        return None, None

    transform = build_test_transform(cfg)
    result = {}

    for mode in ("low", "high"):
        dataset = SNRDataset(base_dataset, snr_threshold, mode, transform)
        loader = build_dataloader(
            dataset=dataset,
            batch_size=int(cfg.get("batch_size", 16)),
            shuffle=False,
        )
        result[mode] = evaluate(model, loader, device, int(cfg.get("num_classes", 2)), threshold)

    return result, cfg


def summarize_c(c_value, result_roots, dataset, device, evaluator):
    results = {}
    run_dirs = iter_run_dirs_for_c(result_roots, c_value)

    print(f"\n[c{c_value}] run dirs: {len(run_dirs)}")
    for idx, run_dir in enumerate(run_dirs, start=1):
        result, cfg = evaluator(run_dir)
        if result is None:
            print(f"  [{idx}/{len(run_dirs)}] missing best_model.pth: {run_dir}")
            continue

        group = config_group_name(cfg)
        results.setdefault(group, []).append(result)
        print(f"  [{idx}/{len(run_dirs)}] {group}/{run_dir.name}")

    return {key: results[key] for key in sorted(results)}


def load_test_dataset(data_dir):
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    return datasets.ImageFolder(root=str(data_dir))


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"[SAVED] {path}")
