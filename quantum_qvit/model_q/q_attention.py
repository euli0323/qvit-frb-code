import logging
from pathlib import Path
from typing import Literal
from math import log2

import torch
from torch import nn

from .q_layer import QuantumLayer

'量子多头自注意力'
class Q_Attention(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 index_qubit_end:int, 
                 num_qlayers: int,
                 type_qc: Literal['z', 'zy', 'zyz'],
                 method_encoding: Literal['angle', 'amplitude'],
                 pro_attn_drop: float = 0.1,
                 #pro_linear_drop: float = 0.1,
                 device_q: str = "default.qubit",
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
                 
        super().__init__()

        '检查输出的单个token的嵌入维度是否可以被num_heads整除'
        assert dim_emb % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.scale = self.dim_head ** (-0.5)
        self.method_encoding = method_encoding

        '根据编码方式计算比特数'
        if method_encoding == 'angle':
            self.num_qubits = self.dim_head   # 角度编码直接用原维度

        elif method_encoding == 'amplitude':
            if self.dim_head <= 0 or (self.dim_head & (self.dim_head - 1)) != 0:
                raise ValueError(f"单个头的嵌入维度{self.dim_head}不是2的幂")
            self.num_qubits = int(log2(self.dim_head))
            self.linear = nn.Linear(self.num_qubits, self.dim_head)

        '分别为Q、K、V定义量子投影层'
        self.q_proj = QuantumLayer(num_qubits = self.num_qubits, 
                                   index_qubit_end = index_qubit_end,
                                   num_qlayers = num_qlayers,
                                   type_qc = type_qc,
                                   method_encoding = method_encoding, 
                                   qdevice = device_q,
                                   use_real = use_real,
                                   api = api,
                                   device_real = device_real,
                                   type_qkv = 'q',
                                   id_encoder=id_encoder,
                                   need_save_res = need_save_res,
                                   path_base = path_base,
                                   logger = logger,
                                   real_qc_recorder = real_qc_recorder,
                                   sim_qc_recorder = sim_qc_recorder,
                                   wait_timeout_seconds = wait_timeout_seconds)

        self.k_proj = QuantumLayer(num_qubits = self.num_qubits, 
                                   index_qubit_end = index_qubit_end,
                                   num_qlayers = num_qlayers,
                                   type_qc = type_qc,
                                   method_encoding = method_encoding, 
                                   qdevice = device_q,
                                   use_real = use_real,
                                   api = api,
                                   device_real = device_real,
                                   type_qkv = 'k',
                                   id_encoder=id_encoder,
                                   need_save_res = need_save_res,
                                   path_base = path_base,
                                   logger = logger,
                                   real_qc_recorder = real_qc_recorder,
                                   sim_qc_recorder = sim_qc_recorder,
                                   wait_timeout_seconds = wait_timeout_seconds)

        self.v_proj = QuantumLayer(num_qubits = self.num_qubits, 
                                   index_qubit_end = index_qubit_end,
                                   num_qlayers = num_qlayers,
                                   type_qc = type_qc,
                                   method_encoding = method_encoding, 
                                   qdevice = device_q,
                                   use_real = use_real,
                                   api = api,
                                   device_real = device_real,
                                   type_qkv = 'v',
                                   id_encoder=id_encoder,
                                   need_save_res = need_save_res,
                                   path_base = path_base,
                                   logger = logger,
                                   real_qc_recorder = real_qc_recorder,
                                   sim_qc_recorder = sim_qc_recorder,
                                   wait_timeout_seconds = wait_timeout_seconds)

        self.attn_drop = nn.Dropout(pro_attn_drop)
        self.o_proj = nn.Linear(dim_emb, dim_emb, bias=False)

        # self.proj_drop = nn.Dropout(pro_linear_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '[batch_size, num_patches + 1, total_embed_dim]'
        B, N, dim_emb = x.shape

        '分开成多个head, 并交换1维和2维的顺序,变为:(B, num_heads, N, dim_head)'
        x = x.reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)

        '用量子线路计算q, k, v'
        # q, k, v = [proj(x) for proj, x in zip([self.q_proj, self.k_proj, self.v_proj], [x, x, x])]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        '如果是幅度编码,需要把比特数恢复成dim_head'
        if self.method_encoding=='amplitude':
            q = self.linear(q)
            k = self.linear(k)
            v = self.linear(v)

        '计算注意力得分+归一化'
        attn = (q @ k.transpose(-2, -1)) * self.scale

        '计算注意力权重矩阵'
        attn = attn.softmax(dim=-1)

        '对权重矩阵随机dropout'
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, dim_emb)
        x = self.o_proj(x)
        # x = self.proj_drop(x)

        return x
    
if __name__ == "__main__":
    '参数'
    batch_size = 2
    num_tokens = 16
    dim_emb = 12

    '输入'
    x = torch.randn(batch_size, num_tokens, dim_emb)
    'api'
    api = 'V7MvvZU:DVxdDyUx7rEDrzsQnSTcHJHi2PsGUpW4zqi/14O4hENxhENvZEP5l{O4d{O4FkPjBIfmKDMjZEO7REO7FUNhNENuRENuZkNxJkJ7JDeimnJtBkPjxX[3WHcjxjJu:3ZvNkOyBVbtWY[gSndiincwWHcjpkJzW3d2Kzf'
    
    model1 = Q_Attention(dim_emb = dim_emb, 
                        num_heads = 3,
                        num_qlayers = 2,
                        method_encoding = 'angle',
                        use_real=True,
                        api = api,
                        device_real = None)

    model1.eval()

    '输出'
    with torch.no_grad():
        y1 = model1(x)
    
    print(f"y1的形状是:{y1.shape}")
    # print(f"y2的形状是:{y2.shape}")
    print(f"q_proj的形状:{model1.q_proj.state_dict()['qc.weights'].shape}")
