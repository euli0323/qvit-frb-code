from typing import Literal
from pathlib import Path
import logging

import torch
from torch import nn

from .q_layer import QuantumLayer
from .q_attention import Q_Attention

'样本随机丢弃'
def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 随机drop一个完整的block，
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self, 
                 dim_emb: int, 
                 dim_hidden: int, 
                 act_layer: type(nn.Module)= nn.GELU,
                 pro_drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim_emb, dim_hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim_hidden, dim_emb)
        self.drop = nn.Dropout(pro_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class QMLP(nn.Module):
    def __init__(self, 
                 dim_emb: int, 
                 num_qlayers: int, 
                 method_encoding: Literal['angle', 'amplitude'] = 'angle',
                 act_layer: type(nn.Module)= nn.GELU,
                 pro_drop: float = 0.,
                 device_q: str = "default.qubit"):
        super().__init__()
        self.fc = QuantumLayer(num_qubits = dim_emb, 
                               method_encoding=method_encoding, 
                               num_qlayers = num_qlayers,
                               qdevice = device_q)
        self.act = act_layer()
        self.drop = nn.Dropout(pro_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)

        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

'Q-Transformer编码器层(MLP)'
class Q_Transformer_Encoder_MLP(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 index_qubit_end: int, 
                 num_qlayers: int,
                 type_qc: Literal['z', 'zy', 'zyz'],
                 method_encoding:Literal['angle', 'amplitude'],
                 mlp_ratio = 4.0,
                 pro_linear_drop = 0.0,
                 pro_attn_drop = 0.0,
                 pro_path_drop = 0.0,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 use_real: bool = False,
                 api: str = None,
                 device_real: str = None,
                 id_encoder: int = None,
                 need_save_res: bool = False,
                 path_base: Path = None,
                 logger: logging.Logger = None,
                 real_qc_recorder = None,
                 sim_qc_recorder = None,
                 wait_timeout_seconds: float | None = 300):

        super(Q_Transformer_Encoder_MLP, self).__init__()
        self.norm1 = norm_layer(dim_emb)
        self.attn = Q_Attention(dim_emb = dim_emb,
                                num_heads = num_heads,
                                index_qubit_end = index_qubit_end,
                                num_qlayers = num_qlayers,
                                type_qc = type_qc,
                                method_encoding = method_encoding,
                                pro_attn_drop = pro_attn_drop,
                                use_real = use_real,
                                api = api,
                                device_real = device_real,
                                id_encoder = id_encoder,
                                need_save_res = need_save_res,
                                path_base = path_base,
                                logger = logger,
                                real_qc_recorder = real_qc_recorder,
                                sim_qc_recorder = sim_qc_recorder,
                                wait_timeout_seconds = wait_timeout_seconds)

        self.drop_path = DropPath(pro_path_drop) if pro_path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_emb)
        dim_hidden_mlp = int(dim_emb * mlp_ratio)
        self.mlp = MLP(dim_emb = dim_emb, 
                       dim_hidden = dim_hidden_mlp, 
                       act_layer = act_layer, 
                       pro_drop = pro_linear_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

'Q-Transformer编码器层(Q-Encoder)'
class Q_Transformer_Encoder_QMLP(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 num_qlayers: int,
                 method_encoding:Literal['angle', 'amplitude'],
                 pro_linear_drop = 0.0,
                 pro_attn_drop = 0.0,
                 pro_path_drop = 0.0,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
                 
        super(Q_Transformer_Encoder_QMLP, self).__init__()
        self.norm1 = norm_layer(dim_emb)
        self.attn = Q_Attention(dim_emb = dim_emb,
                                num_heads = num_heads,
                                num_qlayers = num_qlayers,
                                method_encoding = method_encoding,
                                pro_attn_drop = pro_attn_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(pro_path_drop) if pro_path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_emb)
        self.qmlp = QMLP(dim_emb = dim_emb, 
                         num_qlayers = num_qlayers,
                       method_encoding = method_encoding,
                       act_layer = act_layer, 
                       pro_drop = pro_linear_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.qmlp(self.norm2(x)))

        return x

if __name__ == "__main__":
    # 测试用
    batch_size = 2
    seq_len = 8
    dim_emb = 16
    num_heads = 2
    num_qlayers = 2

    # 随机输入
    x = torch.randn(batch_size, seq_len, dim_emb)

    # 初始化编码器
    encoder1 = Q_Transformer_Encoder_MLP(
        dim_emb=dim_emb,
        num_heads=num_heads,
        num_qlayers = num_qlayers,
        method_encoding='angle',
        pro_linear_drop=0.1,
        pro_attn_drop=0.1,
        pro_path_drop=0.1)

    encoder2 = Q_Transformer_Encoder_QMLP(
        dim_emb=dim_emb,
        num_heads=num_heads,
        num_qlayers = num_qlayers,
        method_encoding='angle',
        pro_linear_drop=0.1,
        pro_attn_drop=0.1,
        pro_path_drop=0.1)

    # 前向传播
    out1 = encoder1(x)
    out2 = encoder2(x)
    print("输入形状:", x.shape)
    print("加mlp的输出形状:", out1.shape)
    print("加qmlp的输出形状:", out2.shape)
    print(f"encoder1的q_proj的形状:{encoder1.attn.q_proj.state_dict()['qc.weights'].shape}")
    print(f"encoder2的q_proj的形状:{encoder2.attn.q_proj.state_dict()['qc.weights'].shape}")
