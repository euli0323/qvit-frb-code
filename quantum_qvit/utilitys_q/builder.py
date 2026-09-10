'妯″瀷 / loss / optimizer / scheduler 鏋勫缓'

import torch.nn as nn
import torch.optim as optim
import math

from model_q.q_model import QViT
from utilitys_q.loss_function import (
    FocalLoss, F1ScoreLoss, F2ScoreLoss, CombinedLoss
)

def build_model(config, device):
    model = QViT(
        size_image=config['size_image'],
        size_windows=config['size_windows'],
        num_channels=config['num_channels'],
        num_classes=config['num_classes'],
        dim_emb=config['dim_emb'],
        index_qubit_end = config['index_qubit_end'],
        num_qlayers=config['num_qlayers'],
        type_qc = config['type_qc'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        pro_attn_drop=config['pro_attn_drop'],
        pro_linear_drop=config['pro_linear_drop'],
        pro_path_drop=config.get('pro_path_drop', 0.0),
        pos_mode='learnable',
        patch_embed=config.get('patch_embed', 'proj'),
        stem_channels=int(config.get('stem_channels', 32)),
        stem_layers=config.get('stem_layers', 2),
        stem_dilation=config.get('stem_dilation', 1),
        method_encoding=config['method_encoding']
    )
    return model.to(device)


def build_loss(config, class_weights, device):

    name = config['loss_function']

    if name == 'FocalLoss':
        return FocalLoss(
            num_classes=2,
            alpha=[class_weights[0].item(), class_weights[1].item()],
            gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config['label_smoothing']
        ).to(device)

    elif name == 'F1Score':
        return F1ScoreLoss(num_classes=2).to(device)

    elif name == 'F2Score':
        return F2ScoreLoss(num_classes=2, beta=2.0).to(device)

    elif name == 'Combined':
        return CombinedLoss(
            num_classes=2,
            loss_type=config.get('combined_loss_type', 'f2'),
            ce_weight=config.get('combined_ce_weight', 0.5),
            f_weight=config.get('combined_f_weight', 0.5),
            class_weights=class_weights
        ).to(device)

    else:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config['label_smoothing']
        ).to(device)


def build_optimizer(model, config):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=config["lr"])


def build_scheduler(optimizer, config):

    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return float(epoch + 1) / config['warmup_epochs']

        progress = (epoch - config['warmup_epochs']) / \
                   max(1, (config['num_epochs'] - config['warmup_epochs']))

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
