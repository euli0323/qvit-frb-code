import logging
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_q.q_model import QViT


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(PROJECT_ROOT / ".env")


def read_config(config_path: Path):
    cfg = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                print(f"[warn] skip unparsable config line: {line}")
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if re.match(r"^\(.*\)$", value):
                numbers = re.findall(r"-?\d+\.?\d*", value)
                cfg[key] = tuple(float(x) if "." in x else int(x) for x in numbers)
            elif re.match(r"^-?\d+\.?\d*$", value):
                cfg[key] = float(value) if "." in value else int(value)
            else:
                cfg[key] = value
    return cfg


def build_model_from_config(
    device,
    cfg,
    use_real: bool,
    device_real: str,
    need_save_res: bool,
    path_base: Path,
    logger: logging.Logger,
    real_qc_recorder=None,
    sim_qc_recorder=None,
    api_token: str | None = None,
    wait_timeout_seconds: float | None = 300,
):
    api_to_use = api_token or os.getenv("QUARK_API_TOKEN")
    if use_real and not api_to_use:
        raise ValueError(
            "Missing real-machine API token. Set QUARK_API_TOKEN in QVIT_mix/.env or pass --api-token."
        )

    model = QViT(
        size_image=cfg["size_image"],
        size_windows=cfg["size_windows"],
        num_channels=cfg["num_channels"],
        num_classes=cfg["num_classes"],
        dim_emb=cfg["dim_emb"],
        index_qubit_end=cfg.get("index_qubit_end", None),
        num_qlayers=cfg["num_qlayers"],
        type_qc=cfg.get("type_qc", "zy"),
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        pro_attn_drop=cfg["pro_attn_drop"],
        pro_linear_drop=cfg["pro_linear_drop"],
        pos_mode=cfg.get("pos_mode", "learnable"),
        method_encoding=cfg.get("method_encoding", "angle"),
        use_real=use_real,
        api=api_to_use,
        device_real=device_real,
        need_save_res=need_save_res,
        path_base=path_base,
        logger=logger,
        real_qc_recorder=real_qc_recorder,
        sim_qc_recorder=sim_qc_recorder,
        wait_timeout_seconds=wait_timeout_seconds,
    ).to(device)

    return model


def per_image_zscore(img_tensor):
    mean = img_tensor.mean()
    std = img_tensor.std()
    return (img_tensor - mean) / std if std > 0 else img_tensor - mean


def build_transform(cfg):
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize(cfg["size_image"]),
            transforms.ToTensor(),
            transforms.Lambda(per_image_zscore),
        ]
    )


def build_loader(ds, batch_size=16, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def predict_single(model, image, device, transform=None, class_names=None) -> dict:
    model.eval()

    if transform is not None and not isinstance(image, torch.Tensor):
        image = transform(image)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1).item()

    return {
        "pred_class": class_names[pred_idx] if class_names is not None else pred_idx,
        "probabilities": probs.squeeze(0).cpu().tolist(),
    }
