import argparse
import copy
import json
import logging
import os
import random
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "utilitys_q").exists() else SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utilitys_q.builder import build_loss, build_model, build_optimizer, build_scheduler
from utilitys_q.data_process import build_datasets, build_transforms
CLASS_LABELS = {0: "frb", 1: "fri"}
DEFAULT_TEACHER_CHECKPOINT = "teacher_frb0_yxdata/resnet18_baseline/001_seed142/best_model.pth"


DEFAULT_CONFIG = {
    "patch_embed": "conv_stem",
    "stem_channels": 32,
    "stem_layers": 0,
    "stem_dilation": 1,
    "experiment_name": "qvit_mix2_yxdata_distill",
    "data_dir": "data/yxdata",
    "output_dir": "res_q",
    "seed_base": 42,
    "seeds": 10,
    "save_model": True,
    "save_history": True,
    "num_workers_train": 5,
    "num_workers_val": 2,
    "size_image": [288, 288],
    "size_windows": 48,
    "num_channels": 1,
    "num_classes": 2,
    "dim_emb": 12,
    "num_qubits": 3,
    "num_heads": 4,
    "index_qubit_end": 3,
    "num_qlayers": 0,
    "circuit_depth": 0,
    "type_qc": "zy",
    "depth": 1,
    "mlp_ratio": 6.0,
    "method_encoding": "angle",
    "pro_attn_drop": 0.1,
    "pro_linear_drop": 0.1,
    "pro_path_drop": 0.0,
    "weight_decay": 1e-2,
    "batch_size": 32,
    "num_epochs": 120,
    "optimizer": "AdamW",
    "lr": 4e-4,
    "label_smoothing": 0.005,
    "warmup_epochs": 8,
    "scheduler": "CosineWarmup",
    "loss_function": "FocalLoss",
    "focal_gamma": 2.0,
    "combined_loss_type": "f2",
    "combined_ce_weight": 0.5,
    "combined_f_weight": 0.5,
    "class_weight_scale": 1.0,
    "random_translate": [0.08, 0.08],
    "ema_decay": 0.999,
    "early_stop_patience": 40,
    "teacher_checkpoint": None,
    "distill_alpha": 0.2,
    "distill_temperature": 4.0
}


def parse_value(value):
    if isinstance(value, bool):
        return value
    text = str(value)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in text:
        return [parse_value(item.strip()) for item in text.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_bool_flag(value):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, 1/0, yes/no, or on/off.")


def normalize_config(config):
    config = copy.deepcopy(config)
    for key in ("size_image", "size_windows"):
        value = config[key]
        if isinstance(value, list):
            config[key] = tuple(value)
    if config.get("random_translate") is not None and isinstance(config["random_translate"], list):
        config["random_translate"] = tuple(config["random_translate"])
    if "circuit_depth" in config and config["circuit_depth"] is not None:
        config["num_qlayers"] = int(config["circuit_depth"])
    else:
        config["circuit_depth"] = int(config["num_qlayers"])
    config["depth"] = 1
    if "num_qubits" in config and config["num_qubits"] is not None:
        config["num_qubits"] = int(config["num_qubits"])
        config["dim_emb"] = int(config["num_qubits"]) * int(config["num_heads"])
    else:
        config["num_qubits"] = int(config["dim_emb"]) // int(config["num_heads"])
    if config.get("index_qubit_end") is None:
        config["index_qubit_end"] = int(config["num_qubits"])
    else:
        config["index_qubit_end"] = int(config["index_qubit_end"])
    if int(config["dim_emb"]) % int(config["num_heads"]) != 0:
        raise ValueError("dim_emb must be divisible by num_heads.")
    if int(config["dim_emb"]) != int(config["num_qubits"]) * int(config["num_heads"]):
        raise ValueError("dim_emb must equal num_qubits * num_heads for QVIT_mix2.")
    return config


def load_config(path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8-sig") as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def apply_overrides(config, overrides):
    if len(overrides) % 2 != 0:
        raise ValueError("Overrides must be pairs like --set key value.")
    for key, value in zip(overrides[0::2], overrides[1::2]):
        key = key.replace("-", "_")
        config[key] = parse_value(value)
    return config


def apply_unknown_cli_overrides(config, unknown_args):
    overrides = []
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument: {key}")
        if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("--"):
            raise ValueError(f"Missing value for argument: {key}")
        overrides.extend([key[2:], unknown_args[i + 1]])
        i += 2
    return apply_overrides(config, overrides)


def generate_seeds(seed_base, num_runs):
    rng = np.random.RandomState(seed_base)
    return rng.randint(0, 10**6, size=num_runs).tolist()


def load_checkpoint_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_resnet18(num_channels, num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_teacher(config, device):
    path = config.get("teacher_checkpoint")
    if not path:
        return None
    checkpoint = load_checkpoint_compat(path, device)
    arch = str(checkpoint.get("arch", "vit")).lower()
    t_config = copy.deepcopy(checkpoint["config"])
    if "resnet" in arch:
        teacher = build_resnet18(t_config["num_channels"], t_config["num_classes"])
        teacher.load_state_dict(checkpoint["model_state_dict"])
    else:
        teacher = build_model(normalize_config(t_config), device)
        teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def distillation_loss(student_logits, teacher_logits, temperature):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ModelEma:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for key, value in self.module.state_dict().items():
            source = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def setup_logger(run_dir):
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def count_class_weights(train_dir, device, scale):
    dir0 = train_dir / "0"
    dir1 = train_dir / "1"
    size0 = sum(1 for f in dir0.iterdir() if f.is_file()) if dir0.exists() else 0
    size1 = sum(1 for f in dir1.iterdir() if f.is_file()) if dir1.exists() else 0
    if size0 == 0 or size1 == 0:
        raise ValueError(f"Empty class directory: class0={size0}, class1={size1}")
    total = size0 + size1
    weights = torch.tensor(
        [total / (2 * size0) * scale, total / (2 * size1) * scale],
        dtype=torch.float32,
        device=device,
    )
    return weights, {"class0_frb": size0, "class1_fri": size1}


def validate_class_mapping(datasets):
    expected = {"0": 0, "1": 1}
    split_names = ("train", "val", "test")
    mappings = {}
    for split_name, dataset in zip(split_names, datasets):
        class_to_idx = getattr(dataset, "class_to_idx", None)
        mappings[split_name] = class_to_idx
        if class_to_idx != expected:
            raise ValueError(
                f"{split_name} class_to_idx must be {expected} for 0=frb and 1=fri, got {class_to_idx}"
            )
    return mappings


def model_param_stats(model):
    total = 0
    trainable = 0
    quantum = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if "qc" in name:
            quantum += n
    return {
        "total": total,
        "trainable": trainable,
        "quantum": quantum,
        "classical": total - quantum,
    }


def fbeta_for_class(y_true, y_pred, positive_label=0, beta=2.0, epsilon=1e-7):
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
    return float(fbeta), float(recall)


def metrics_from_frb_probs(y_true, prob_frb, threshold):
    y_pred = np.where(prob_frb >= threshold, 0, 1).astype(np.int64)
    acc = 100.0 * float((y_pred == y_true).mean())
    f2, recall = fbeta_for_class(y_true, y_pred, positive_label=0, beta=2.0)
    return acc, f2, recall


def tune_threshold(y_true, prob_frb):
    best_thr = 0.5
    best_f2 = -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        _, f2, _ = metrics_from_frb_probs(y_true, prob_frb, float(thr))
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)
    return best_thr


def format_metrics(prefix, metrics):
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"acc={metrics['acc']:.2f}% "
        f"recall={metrics['recall'] * 100.0:.2f}% "
        f"f2={metrics['f2'] * 100.0:.2f}%"
    )


def save_training_curves(history, path, logger):
    if plt is None:
        logger.warning("matplotlib is not installed; skip saving training curves png")
        return
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    series = [
        ("loss", [item["train"]["loss"] for item in history], [item["val"]["loss"] for item in history]),
        ("acc", [item["train"]["acc"] for item in history], [item["val"]["acc"] for item in history]),
        ("recall", [item["train"]["recall"] * 100.0 for item in history], [item["val"]["recall"] * 100.0 for item in history]),
        ("f2", [item["train"]["f2"] * 100.0 for item in history], [item["val"]["f2"] * 100.0 for item in history]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    for ax, (title, train_values, val_values) in zip(axes.ravel(), series):
        ax.plot(epochs, train_values, label="train", linewidth=1.8)
        ax.plot(epochs, val_values, label="val", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_epoch(model, loader, criterion, device, fixed_threshold=None):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probs_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs_all.append(torch.softmax(outputs, dim=1)[:, 0].cpu())
            labels_all.append(labels.cpu())
    y_prob = torch.cat(probs_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    threshold = fixed_threshold if fixed_threshold is not None else tune_threshold(y_true, y_prob)
    acc, f2, recall = metrics_from_frb_probs(y_true, y_prob, threshold)
    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": acc,
        "f2": f2,
        "recall": recall,
        "threshold": threshold,
    }


def train_one_run(config, run_id, seed, run_dir, datasets, device):
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    setup_seed(seed)

    train_dataset, val_dataset, test_dataset = datasets
    class_weights, class_counts = count_class_weights(
        Path(config["data_dir"]) / "train", device, config["class_weight_scale"]
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers_train"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers_train"] > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers_val"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers_val"] > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers_val"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers_val"] > 0,
    )

    model = build_model(config, device)
    param_stats = model_param_stats(model)
    criterion = build_loss(config, class_weights, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    ema = ModelEma(model, config["ema_decay"]) if config.get("ema_decay", 0.0) > 0 else None

    history = []
    best = {
        "epoch": 0,
        "val_acc": 0.0,
        "val_recall": 0.0,
        "val_f2": 0.0,
        "val_loss": None,
        "val_threshold": 0.5,
    }
    epochs_no_improve = 0
    patience = int(config.get("early_stop_patience", 0) or 0)
    best_snapshot_path = None
    start = time.time()

    logger.info("run_id=%s seed=%s device=%s", run_id, seed, device)
    logger.info("params=%s", param_stats)
    logger.info("class_counts=%s class_weights=%s", class_counts, class_weights.detach().cpu().tolist())
    logger.info("class_labels=%s", CLASS_LABELS)
    logger.info("ema_decay=%s early_stop_patience=%s", config.get("ema_decay", 0.0), patience)

    teacher = load_teacher(config, device)
    kd_alpha = float(config.get("distill_alpha", 0.3))
    kd_temperature = float(config.get("distill_temperature", 4.0))
    if teacher is not None:
        logger.info(
            "distillation enabled: teacher=%s alpha(hard)=%s T=%s",
            config["teacher_checkpoint"], kd_alpha, kd_temperature,
        )

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_kd_loss = 0.0
        train_total = 0
        train_correct = 0
        train_labels_all = []
        train_probs_all = []
        pbar = tqdm(
            train_loader,
            desc=f"run {run_id} seed {seed} epoch {epoch}/{config['num_epochs']}",
            leave=True,
        )
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            hard_loss = criterion(outputs, labels)
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                kd_loss = distillation_loss(outputs, teacher_outputs, kd_temperature)
                loss = kd_alpha * hard_loss + (1.0 - kd_alpha) * kd_loss
                train_kd_loss += kd_loss.item()
            else:
                loss = hard_loss
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()
            train_probs_all.append(torch.softmax(outputs.detach(), dim=1)[:, 0].cpu())
            train_labels_all.append(labels.detach().cpu())

        eval_model = ema.module if ema is not None else model
        val = evaluate_epoch(eval_model, val_loader, criterion, device)
        scheduler.step()
        train_prob = torch.cat(train_probs_all).numpy()
        train_true = torch.cat(train_labels_all).numpy()
        train_acc, train_f2, train_recall = metrics_from_frb_probs(train_true, train_prob, 0.5)
        train = {
            "loss": train_loss / max(1, len(train_loader)),
            "acc": train_acc,
            "recall": train_recall,
            "f2": train_f2,
            "threshold": 0.5,
        }
        if teacher is not None:
            train["kd_loss"] = train_kd_loss / max(1, len(train_loader))
        epoch_item = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train,
            "val": val,
        }
        history.append(epoch_item)
        logger.info("epoch=%03d", epoch)
        logger.info(format_metrics("train", train))
        logger.info(format_metrics("val", val))

        if val["f2"] > best["val_f2"]:
            best = {
                "epoch": epoch,
                "val_acc": val["acc"],
                "val_recall": val["recall"],
                "val_f2": val["f2"],
                "val_loss": val["loss"],
                "val_threshold": val["threshold"],
            }
            epochs_no_improve = 0
            if config["save_model"]:
                snapshot_path = run_dir / (
                    f"best_model_acc{val['acc']:.2f}_recall{val['recall']:.4f}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "seed": seed,
                        "config": config,
                        "model_state_dict": eval_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best": best,
                        "is_ema": ema is not None,
                    },
                    snapshot_path,
                )
                shutil.copyfile(snapshot_path, run_dir / "best_model.pth")
                if best_snapshot_path is not None and best_snapshot_path.exists():
                    best_snapshot_path.unlink()
                best_snapshot_path = snapshot_path
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "early stopping at epoch %d (no val f2 improvement for %d epochs)",
                    epoch,
                    patience,
                )
                break

    best_model_path = run_dir / "best_model.pth"
    if config["save_model"] and best_model_path.exists():
        checkpoint = load_checkpoint_compat(best_model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("loaded best model from epoch %s for test evaluation", best["epoch"])
        final_val = evaluate_epoch(
            model, val_loader, criterion, device, fixed_threshold=best["val_threshold"]
        )
    else:
        final_val = history[-1]["val"]

    test = evaluate_epoch(model, test_loader, criterion, device, fixed_threshold=best["val_threshold"])
    elapsed = time.time() - start
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "run_id": run_id,
        "seed": seed,
        "status": "finished",
        "started_at": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "host": socket.gethostname(),
        "config": config,
        "data": {
            "data_dir": config["data_dir"],
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "num_test": len(test_dataset),
            "class_counts_train": class_counts,
            "class_weights": class_weights.detach().cpu().tolist(),
        },
        "params": param_stats,
        "best": best,
        "final": {
            "train": history[-1]["train"],
            "val": final_val,
            "test": test,
        },
    }
    if config["save_history"]:
        result["history"] = history
    result["class_labels"] = CLASS_LABELS

    save_training_curves(history, run_dir / "training_curves.png", logger)
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def build_run_name(config):
    dim_head = int(config["dim_emb"]) // int(config["num_heads"])
    return f"q{dim_head}_circuit_depth{config['circuit_depth']}_end{config['index_qubit_end']}"


def build_output_root(config):
    return Path(config["output_dir"]) / build_run_name(config)


def main():
    parser = argparse.ArgumentParser(description="Cloud runner for QVIT_mix2 repeated-seed experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--set", nargs="*", default=[], help="Override config entries: --set key value key value")
    parser.add_argument("--seeds-list", type=str, default=None, help="Comma-separated explicit seeds.")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override output_dir after normalization (for testing).")
    parser.add_argument("--num-qubits", type=int, default=None,
                        help="Number of qubits per attention head. num_heads defaults to 4.")
    parser.add_argument("--circuit-depth", type=int, default=None,
                        help="Quantum circuit depth, mapped to num_qlayers.")
    parser.add_argument("--use-teacher", type=parse_bool_flag, default=None,
                        help="Whether to use a teacher checkpoint for distillation.")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Path to a teacher checkpoint (e.g. ResNet-18 best_model.pth) for distillation.")
    args, unknown_args = parser.parse_known_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    config = apply_unknown_cli_overrides(config, unknown_args)
    if args.num_qubits is not None:
        config["num_qubits"] = args.num_qubits
        config["index_qubit_end"] = args.num_qubits
    if args.circuit_depth is not None:
        config["circuit_depth"] = args.circuit_depth
    if args.use_teacher is True:
        config["teacher_checkpoint"] = args.teacher or DEFAULT_TEACHER_CHECKPOINT
        config["output_dir"] = "res_q_distill"
    elif args.use_teacher is False:
        config["teacher_checkpoint"] = None
    elif args.teacher:
        config["teacher_checkpoint"] = args.teacher
    config = normalize_config(config)
    if args.output_root:
        config["output_dir"] = args.output_root

    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_train, transform_val = build_transforms(config)
    datasets = build_datasets(data_dir, transform_train, transform_val)
    class_mappings = validate_class_mapping(datasets)

    if args.seeds_list:
        seeds = [int(item.strip()) for item in args.seeds_list.split(",") if item.strip()]
    else:
        seeds = generate_seeds(config["seed_base"], config["seeds"])

    root = build_output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    results = []
    for run_id, seed in enumerate(seeds, start=1):
        run_dir = root / f"{run_id:03d}_seed{seed}"
        try:
            results.append(train_one_run(config, run_id, seed, run_dir, datasets, device))
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "experiment_name": config["experiment_name"],
                "run_id": run_id,
                "seed": seed,
                "status": "failed",
                "error": repr(exc),
                "config": config,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "run.json", "w", encoding="utf-8") as f:
                json.dump(failure, f, ensure_ascii=False, indent=2)
            raise

    best_f2 = [item["best"]["val_f2"] for item in results]
    best_acc = [item["best"]["val_acc"] for item in results]
    summary = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "finished",
        "num_runs": len(results),
        "output_dir": str(root),
        "config": config,
        "class_labels": CLASS_LABELS,
        "class_mappings": class_mappings,
        "runs": [
            {
                "run_id": item["run_id"],
                "seed": item["seed"],
                "status": item["status"],
                "best": item["best"],
                "final": item["final"],
                "params": item["params"],
                "run_json": str(root / f"{item['run_id']:03d}_seed{item['seed']}" / "run.json"),
            }
            for item in results
        ],
        "aggregate": {
            "best_val_f2_mean": float(np.mean(best_f2)),
            "best_val_f2_std": float(np.std(best_f2, ddof=1)) if len(best_f2) > 1 else 0.0,
            "best_val_acc_mean": float(np.mean(best_acc)),
            "best_val_acc_std": float(np.std(best_acc, ddof=1)) if len(best_acc) > 1 else 0.0,
        },
    }
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
