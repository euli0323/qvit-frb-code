import math

import torch
from torch import nn

class Position_Embedding(nn.Module):
    def __init__(self, d_model: int, n_pos: int, mode: str = 'learnable'):
        super().__init__()
        assert mode in ['learnable', 'sinusoidal']
        self.d_model = d_model
        self.mode = mode
        self.n_pos = n_pos  #保存 token 数，保证一致性

        # 分类 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if mode == 'learnable':
            # (1, n_pos+1, d_model) —— 包含 cls_token
            self.pos_embed = nn.Parameter(torch.zeros(1, n_pos + 1, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # 正余弦编码缓存
            pe = self._build_sinusoidal(n_pos + 1, d_model)
            self.register_buffer('sin_cache', pe, persistent=False)

    @staticmethod
    def _build_sinusoidal(n_pos: int, d_model: int, device=None, dtype=None) -> torch.Tensor:
        position = torch.arange(n_pos, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(n_pos, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, n_pos, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.d_model, f"dim mismatch: got {D}, expected {self.d_model}"
        assert N == self.n_pos, f"token mismatch: got {N}, expected {self.n_pos}"

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)          # (B, N+1, D)

        if self.mode == 'learnable':
            x = x + self.pos_embed
        else:
            x = x + self.sin_cache

        return x

if __name__ == "__main__":
    '设定参数'
    batch_size = 2
    num_tokens = 500     # patch数量
    d_model = 256        # emb维度

    '构造输入'
    x = torch.randn(batch_size, num_tokens, d_model)

    print("===== 测试 Learnable Position Embedding =====")
    pe1 = Position_Embedding(d_model=d_model, n_pos=num_tokens, mode='learnable')
    out1 = pe1(x)

    print("输出形状:", out1.shape)               # (2, 501, 256)
    print("cls token 是否正常加入:", torch.allclose(out1[:,0], pe1.cls_token.expand(batch_size, -1, -1).squeeze(1), atol=1e-4))
    print()

    print("===== 测试 Sinusoidal Position Embedding =====")
    pe2 = Position_Embedding(d_model=d_model, n_pos=num_tokens, mode='sinusoidal')
    out2 = pe2(x)

    print("输出形状:", out2.shape)               # (2, 501, 256)
    print("sin cache 形状:", pe2.sin_cache.shape) # (1, 501, 256)
    print("第0个位置是否为sinusoidal编码:", out2[0, 0, :5])  # 打印前 5 个数看看
    print()

    '检查两种模式不一样'
    print("learnable 与 sinusoidal 是否不同:", not torch.allclose(out1, out2))