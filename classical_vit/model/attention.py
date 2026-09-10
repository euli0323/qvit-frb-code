import torch
from torch import nn


class HeadwiseMLPProjection(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_head_before_qkv_proj: int,
                 hidden_dim: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head_before_qkv_proj = dim_head_before_qkv_proj
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Parameter(torch.empty(num_heads, dim_head_before_qkv_proj, hidden_dim))
        self.fc2 = nn.Parameter(torch.empty(num_heads, hidden_dim, dim_head_before_qkv_proj))
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.fc1, std=0.02)
        nn.init.trunc_normal_(self.fc2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bnhd,hdm->bnhm', x, self.fc1)
        x = self.act(x)
        x = torch.einsum('bnhm,hmd->bnhd', x, self.fc2)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 dim_head_before_qkv_proj: int,
                 dim_head_after_qkv_proj: int = None,
                 pro_attn_drop: float = 0.1,
                 qkv_projection: str = 'linear',
                 qkv_mlp_hidden_dim: int = 3):

        super().__init__()

        self.num_heads = num_heads
        self.dim_head_before_qkv_proj = dim_head_before_qkv_proj
        self.qkv_projection = qkv_projection

        if qkv_projection == 'linear':
            if dim_head_after_qkv_proj is None:
                dim_head_after_qkv_proj = dim_head_before_qkv_proj
            self.dim_head_after_qkv_proj = dim_head_after_qkv_proj
            self.dim_inner = num_heads * dim_head_after_qkv_proj
            self.scale = self.dim_head_after_qkv_proj ** (-0.5)
            self.qkv = nn.Linear(dim_emb, self.dim_inner * 3, bias=False)
        elif qkv_projection == 'mlp':
            self.dim_inner = num_heads * dim_head_before_qkv_proj
            if dim_emb != self.dim_inner:
                raise ValueError("qkv_projection='mlp' requires dim_emb == num_heads * dim_head_before_qkv_proj")
            self.dim_head_after_qkv_proj = dim_head_before_qkv_proj
            self.scale = self.dim_head_after_qkv_proj ** (-0.5)
            self.q_proj = HeadwiseMLPProjection(num_heads, dim_head_before_qkv_proj, qkv_mlp_hidden_dim)
            self.k_proj = HeadwiseMLPProjection(num_heads, dim_head_before_qkv_proj, qkv_mlp_hidden_dim)
            self.v_proj = HeadwiseMLPProjection(num_heads, dim_head_before_qkv_proj, qkv_mlp_hidden_dim)
        else:
            raise ValueError(f"Unsupported qkv_projection: {qkv_projection}")

        self.attn_drop = nn.Dropout(pro_attn_drop)
        self.o_proj = nn.Linear(self.dim_inner, dim_emb, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, dim_emb = x.shape

        if self.qkv_projection == 'linear':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head_after_qkv_proj).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            x_heads = x.reshape(B, N, self.num_heads, self.dim_head_before_qkv_proj)
            q = self.q_proj(x_heads).transpose(1, 2)
            k = self.k_proj(x_heads).transpose(1, 2)
            v = self.v_proj(x_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_inner)
        x = self.o_proj(x)

        return x


if __name__ == "__main__":
    batch_size = 2
    num_tokens = 5
    num_heads = 4
    dim_head_before_qkv_proj = 5
    dim_emb = 20

    x1 = torch.randn(batch_size, num_tokens, dim_emb)
    print(f"x1 shape: {x1.shape}")
    model1 = Attention(dim_emb=dim_emb,
                       num_heads=num_heads,
                       dim_head_before_qkv_proj=dim_head_before_qkv_proj,
                       qkv_projection='mlp')

    y1 = model1(x1)
    print(f"y1 shape: {y1.shape}")
