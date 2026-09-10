import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from q_model import QViT


# ======================================================
# 1. 数据集
# ======================================================
class FRBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        for label in ['0', '1']:
            folder = os.path.join(root_dir, label)
            for f in os.listdir(folder):
                self.samples.append(os.path.join(folder, f))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ======================================================
# 2. 创建模型
# ======================================================
def build_model(device):
    model = QViT(
        size_image=(680, 850),
        size_windows=170,
        num_channels=1,
        num_classes=2,
        dim_emb=10,
        depth=3,
        num_heads=2,
        mlp_ratio=4.0,
        pro_attn_drop=0.1,
        pro_linear_drop=0.1,
        pro_path_drop=0.0,
        pos_mode='learnable',
        method_encoding='angle'
    )
    model.to(device)
    return model


# ======================================================
# 3. 加载权重
# ======================================================
# def load_weights(model, ckpt_path):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     dict_par = ckpt["model_state_dict"]
    
    
#     for i, encoder in enumerate(model.encoders):
#         q_key = f'encoders.{i}.attn.q_proj.linear.weights'
#         k_key = f'encoders.{i}.attn.k_proj.linear.weights'
#         v_key = f'encoders.{i}.attn.v_proj.linear.weights'
        
#         if q_key in dict_par and k_key in dict_par and v_key in dict_par:
#             with torch.no_grad():
#                 encoder.attn.q_proj.params.copy_(dict_par[q_key])
#                 encoder.attn.k_proj.params.copy_(dict_par[k_key])
#                 encoder.attn.v_proj.params.copy_(dict_par[v_key])
#             print(f"[权重] encoder {i} 的 Q/K/V 已迁移")
#         else:
#             print(f"[警告] encoder {i} 缺少部分权重，未全部迁移")

# def load_weights(model, ckpt_path):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     dict_par = ckpt["model_state_dict"]

#     # 先加载除 Q/K/V 外的权重
#     filtered_dict = {
#         k: v for k, v in dict_par.items()
#         if not any(qkv in k for qkv in ["q_proj.linear.weights",
#                                         "k_proj.linear.weights",
#                                         "v_proj.linear.weights"])
#     }
#     model.load_state_dict(filtered_dict, strict=False)
#     print("[权重] 非 Q/K/V 权重已加载")

#     # 再手动加载 Q/K/V 权重
#     for i, encoder in enumerate(model.encoders):
#         for name in ["q_proj", "k_proj", "v_proj"]:
#             key = f'encoders.{i}.attn.{name}.linear.weights'
#             if key in dict_par:
#                 with torch.no_grad():
#                     getattr(encoder.attn, name).params.copy_(dict_par[key])
#                 print(f"[权重] encoder {i} 的 {name} 已迁移")
#             else:
#                 print(f"[警告] encoder {i} 的 {name} 权重不存在")

# ======================================================
# 4. 数据加载
# ======================================================
def build_loader(test_root):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((680, 850)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = FRBDataset(test_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return loader


# ======================================================
# 5. 推理（返回 accuracy + recall）
# ======================================================
@torch.no_grad()
def evaluate(model, 
             loader, 
             device, 
             num_max:int, 
             num_classes=2):
    model.eval()
    total = 0
    correct = 0

    # 混淆矩阵（2x2）
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # 收集正类概率用于画分布
    all_probs_pos = []
    all_labels = []

    for (i, (images, labels)) in enumerate(loader):
        if i+1>num_max:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        # 混淆矩阵更新
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf[t.long(), p.long()] += 1

        # 收集正类概率（假设类别 1 为“正类”）
        all_probs_pos.append(probs[:, 1].detach().cpu())
        all_labels.append(labels.detach().cpu())

        print(f'第{i+1}批样本已推理完成')

    all_probs_pos = torch.cat(all_probs_pos, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 总体准确率
    overall_acc = correct / total if total > 0 else 0.0

    # 每类准确率（按真实标签计算：该类被判对的比例）
    per_class_acc = {}
    for c in range(num_classes):
        tp = conf[c, c].item()
        tot_c = conf[c, :].sum().item()
        per_class_acc[c] = (tp / tot_c) if tot_c > 0 else 0.0

    return overall_acc, per_class_acc, conf, all_probs_pos, all_labels

# ======================================================
# 6. 主程序
# ======================================================
if __name__ == "__main__":
    raise SystemExit(
        "This module provides inference helper functions. "
        "Use quantum_qvit/scripts/res_test.py for checkpoint evaluation."
    )
