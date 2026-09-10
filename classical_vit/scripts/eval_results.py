import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import load_checkpoint_compat, normalize_config
from utilitys.builder import build_model
from utilitys.data_process import build_transforms

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_BEFORE_VALUES = (3, 4, 5)
DEFAULT_SNR_THRESHOLD = 12.0


def parse_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")


def read_run_payload(run_dir):
    run_json = run_dir / "run.json"
    if run_json.exists():
        return read_json(run_json)
    return {}


def read_run_config(run_dir):
    payload = read_run_payload(run_dir)
    if "config" in payload:
        return normalize_config(payload["config"])

    for config_path in (run_dir / "resolved_config.json", run_dir.parent / "resolved_config.json"):
        if config_path.exists():
            return normalize_config(read_json(config_path))

    raise FileNotFoundError(f"No run.json or resolved_config.json found for {run_dir}")


def config_group_name(config):
    dim_before = int(config["dim_head_before_qkv_proj"])
    num_heads = int(config.get("num_heads", 4))
    emb_dim = dim_before * num_heads
    projection = str(config.get("qkv_projection", "linear")).lower()
    if projection == "linear":
        dim_after = int(config["dim_head_after_qkv_proj"])
        return f"linear_emb{emb_dim}_inner{dim_after * num_heads}"
    hidden = int(config["qkv_mlp_hidden_dim"])
    return f"mlp_emb{emb_dim}_inner{hidden * num_heads}"


def iter_run_dirs(result_roots):
    seen = set()
    for root in result_roots:
        if not root.exists():
            print(f"[WARN] missing result root: {root}")
            continue

        for ckpt_path in sorted(root.rglob("best_model.pth")):
            run_dir = ckpt_path.parent
            resolved = str(run_dir.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield run_dir


def iter_run_dirs_for_before(result_roots, before_value):
    run_dirs = []
    for run_dir in iter_run_dirs(result_roots):
        try:
            config = read_run_config(run_dir)
        except Exception as exc:
            print(f"[WARN] skip {run_dir}: {exc}")
            continue
        if int(config["dim_head_before_qkv_proj"]) == int(before_value):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def load_model(run_dir, device):
    config = read_run_config(run_dir)
    checkpoint = load_checkpoint_compat(run_dir / "best_model.pth", device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = build_model(config, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint if isinstance(checkpoint, dict) else {}


def load_test_dataset(data_dir):
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    dataset = datasets.ImageFolder(root=str(data_dir))
    if dataset.class_to_idx != {"0": 0, "1": 1}:
        raise ValueError(f"class_to_idx must be {{'0': 0, '1': 1}} for 0=FRB and 1=RFI/FRI, got {dataset.class_to_idx}")
    return dataset


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

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def f2_from_counts(tp, fp, fn, epsilon=1e-8):
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f2 = 5.0 * precision * recall / (4.0 * precision + recall + epsilon)
    return precision, recall, f2


@torch.no_grad()
def evaluate_dataset(model, dataset, transform, device, batch_size, threshold=None):
    dataset = copy.copy(dataset)
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    labels_all = []
    probs_all = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs_all.append(torch.softmax(outputs, dim=1)[:, 0].cpu().numpy())
        labels_all.append(labels.numpy())

    if not labels_all:
        return empty_metrics()

    y_true = np.concatenate(labels_all).astype(np.int64)
    prob_frb = np.concatenate(probs_all)
    if threshold is None:
        y_pred = np.where(prob_frb >= 0.5, 0, 1).astype(np.int64)
    else:
        y_pred = np.where(prob_frb >= float(threshold), 0, 1).astype(np.int64)

    tp = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 0) & (y_true != 0)))
    fn = int(np.sum((y_pred != 0) & (y_true == 0)))
    tn = int(np.sum((y_pred != 0) & (y_true != 0)))
    precision, recall, f2 = f2_from_counts(tp, fp, fn)
    total = int(y_true.size)
    acc = (tp + tn) / total if total else 0.0

    return {
        "n": total,
        "acc": round(acc * 100.0, 4),
        "recall": round(recall * 100.0, 4),
        "precision": round(precision * 100.0, 4),
        "f2": round(f2 * 100.0, 4),
        "threshold": threshold if threshold is not None else 0.5,
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "positive_class": "0=FRB",
    }


def empty_metrics():
    return {
        "n": 0,
        "acc": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "f2": 0.0,
        "threshold": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "positive_class": "0=FRB",
    }


def evaluate_run(run_dir, base_dataset, device, snr_threshold):
    model, config, checkpoint = load_model(run_dir, device)
    _, transform_val = build_transforms(config)
    threshold = None
    if isinstance(checkpoint.get("best"), dict):
        threshold = checkpoint["best"].get("val_threshold")
    if threshold is None:
        threshold = read_run_payload(run_dir).get("best", {}).get("val_threshold", 0.5)

    batch_size = int(config.get("batch_size", 32))
    full = evaluate_dataset(model, base_dataset, transform_val, device, batch_size, threshold)
    snr = {}
    for mode in ("low", "high"):
        snr_dataset = SNRDataset(base_dataset, snr_threshold, mode, transform_val)
        snr[mode] = evaluate_dataset(model, snr_dataset, None, device, batch_size, threshold)

    payload = read_run_payload(run_dir)
    return {
        "run_dir": str(run_dir),
        "run_id": payload.get("run_id"),
        "seed": payload.get("seed", checkpoint.get("seed")),
        "group": config_group_name(config),
        "config": {
            "qkv_projection": config.get("qkv_projection"),
            "dim_head_before_qkv_proj": int(config["dim_head_before_qkv_proj"]),
            "dim_head_after_qkv_proj": int(config.get("dim_head_after_qkv_proj", 0)),
            "num_heads": int(config.get("num_heads", 4)),
            "dim_emb": int(config.get("dim_emb", int(config["dim_head_before_qkv_proj"]) * int(config.get("num_heads", 4)))),
            "depth": int(config.get("depth", 1)),
            "size_image": list(config["size_image"]) if isinstance(config["size_image"], tuple) else config["size_image"],
            "size_windows": list(config["size_windows"]) if isinstance(config["size_windows"], tuple) else config["size_windows"],
        },
        "best": payload.get("best", checkpoint.get("best")),
        "full": full,
        "snr12": snr,
    }


def compact_full_result(result):
    return {
        "seed": result["seed"],
        "acc": result["full"]["acc"],
        "recall": result["full"]["recall"],
        "precision": result["full"]["precision"],
        "f2": result["full"]["f2"],
    }


def compact_snr_result(result):
    return {
        "seed": result["seed"],
        "low": {
            "acc": result["snr12"]["low"]["acc"],
            "recall": result["snr12"]["low"]["recall"],
            "precision": result["snr12"]["low"]["precision"],
            "f2": result["snr12"]["low"]["f2"],
        },
        "high": {
            "acc": result["snr12"]["high"]["acc"],
            "recall": result["snr12"]["high"]["recall"],
            "precision": result["snr12"]["high"]["precision"],
            "f2": result["snr12"]["high"]["f2"],
        },
    }


def aggregate_metric(items, section, metric):
    values = [item[section][metric] for item in items if item.get(section) and item[section].get("n", 0) > 0]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
    }


def aggregate_group(items):
    return {
        "num_runs": len(items),
        "full": {
            "acc": aggregate_metric(items, "full", "acc"),
            "recall": aggregate_metric(items, "full", "recall"),
            "f2": aggregate_metric(items, "full", "f2"),
        },
        "snr12_low": {
            "acc": aggregate_snr_metric(items, "low", "acc"),
            "recall": aggregate_snr_metric(items, "low", "recall"),
            "f2": aggregate_snr_metric(items, "low", "f2"),
        },
        "snr12_high": {
            "acc": aggregate_snr_metric(items, "high", "acc"),
            "recall": aggregate_snr_metric(items, "high", "recall"),
            "f2": aggregate_snr_metric(items, "high", "f2"),
        },
    }


def aggregate_snr_metric(items, mode, metric):
    values = [
        item["snr12"][mode][metric]
        for item in items
        if item.get("snr12") and item["snr12"].get(mode, {}).get("n", 0) > 0
    ]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VIT_mix2 checkpoints on full test set and SNR=12 subsets.")
    parser.add_argument("--data-dir", default="data/yxdata/test", help="Path to test split with 0/1 class folders.")
    parser.add_argument(
        "--res-train-dirs",
        nargs="+",
        default=["res_c_linear_qkv"],
        help="One or more checkpoint roots to scan. Pass both cloud project result roots if before=4 is separate.",
    )
    parser.add_argument("--out-dir", default="eval_results", help="Directory for compact per-before JSON files.")
    parser.add_argument("--detail-out-dir", default=None, help="Optional directory for detailed per-before JSON files.")
    parser.add_argument("--before-values", nargs="+", type=int, default=list(DEFAULT_BEFORE_VALUES))
    parser.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = parse_path(args.data_dir)
    result_roots = [parse_path(item) for item in args.res_train_dirs]
    out_dir = parse_path(args.out_dir)

    print(f"[INFO] device: {device}")
    print(f"[INFO] test data: {data_dir}")
    print(f"[INFO] result roots: {[str(path) for path in result_roots]}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] before values: {args.before_values}")
    print(f"[INFO] SNR threshold: {args.snr_threshold}")

    base_dataset = load_test_dataset(data_dir)
    low_count = sum(1 for path, _ in base_dataset.samples if extract_snr(path) < args.snr_threshold)
    high_count = len(base_dataset) - low_count
    print(f"[INFO] test samples: {len(base_dataset)}")
    print(f"[INFO] low SNR samples : {low_count}")
    print(f"[INFO] high SNR samples: {high_count}")

    for before in args.before_values:
        run_dirs = iter_run_dirs_for_before(result_roots, before)
        print(f"\n[c{before}] run dirs: {len(run_dirs)}")

        compact_full = {}
        compact_snr = {}
        detail_groups = {}
        for index, run_dir in enumerate(run_dirs, start=1):
            try:
                result = evaluate_run(run_dir, base_dataset, device, args.snr_threshold)
            except Exception as exc:
                print(f"  [{index}/{len(run_dirs)}] skip {run_dir}: {exc}")
                continue

            group = result["group"]
            compact_full.setdefault(group, []).append(compact_full_result(result))
            compact_snr.setdefault(group, []).append(compact_snr_result(result))
            detail_groups.setdefault(group, []).append(result)
            print(f"  [{index}/{len(run_dirs)}] {group}/{run_dir.name}", flush=True)

        compact_full = {key: compact_full[key] for key in sorted(compact_full)}
        compact_snr = {key: compact_snr[key] for key in sorted(compact_snr)}
        save_json(out_dir / f"res_test_c{before}.json", compact_full)
        save_json(out_dir / f"res_test_snr_c{before}.json", compact_snr)

        if args.detail_out_dir:
            detail_out_dir = parse_path(args.detail_out_dir)
            detail_payload = {
                "schema_version": 1,
                "metric_rule": "0=FRB is the positive class; acc is overall accuracy; recall/F2 are FRB-class metrics.",
                "snr_threshold": args.snr_threshold,
                "data_dir": str(data_dir),
                "result_roots": [str(path) for path in result_roots],
                "dim_head_before_qkv_proj": before,
                "groups": {
                    group: {
                        "aggregate": aggregate_group(items),
                        "runs": items,
                    }
                    for group, items in sorted(detail_groups.items())
                },
            }
            save_json(detail_out_dir / f"vit_mix2_before{before}_detail.json", detail_payload)


if __name__ == "__main__":
    main()
