import math

import torch
from torch import nn

class Position_Embedding(nn.Module):
    def __init__(self, d_model: int, n_pos: int, mode: str = 'learnable'):
        super().__init__()
        assert mode in ['learnable', 'sinusoidal']
        self.d_model = d_model
        self.mode = mode
        self.n_pos = n_pos  #淇濆瓨 token 鏁帮紝淇濊瘉涓€鑷存€?

        # 鍒嗙被 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if mode == 'learnable':
            # (1, n_pos+1, d_model) 鈥斺€?鍖呭惈 cls_token
            self.pos_embed = nn.Parameter(torch.zeros(1, n_pos + 1, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # 姝ｄ綑寮︾紪鐮佺紦瀛?
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
    batch_size = 2
    num_tokens = 500     # patch鏁伴噺
    d_model = 256        # emb缁村害

    x = torch.randn(batch_size, num_tokens, d_model)

    print("===== 娴嬭瘯 Learnable Position Embedding =====")
    pe1 = Position_Embedding(d_model=d_model, n_pos=num_tokens, mode='learnable')
    out1 = pe1(x)

    print("杈撳嚭褰㈢姸:", out1.shape)               # (2, 501, 256)
    print("cls token 鏄惁姝ｅ父鍔犲叆:", torch.allclose(out1[:,0], pe1.cls_token.expand(batch_size, -1, -1).squeeze(1), atol=1e-4))
    print()

    print("===== 娴嬭瘯 Sinusoidal Position Embedding =====")
    pe2 = Position_Embedding(d_model=d_model, n_pos=num_tokens, mode='sinusoidal')
    out2 = pe2(x)

    print("杈撳嚭褰㈢姸:", out2.shape)               # (2, 501, 256)
    print("sin cache 褰㈢姸:", pe2.sin_cache.shape) # (1, 501, 256)
    print("绗?涓綅缃槸鍚︿负sinusoidal缂栫爜:", out2[0, 0, :5])  # 鎵撳嵃鍓?5 涓暟鐪嬬湅
    print()

    print("learnable 涓?sinusoidal 鏄惁涓嶅悓:", not torch.allclose(out1, out2))
