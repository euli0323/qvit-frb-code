# from typing import Literal

import torch
from torch import nn

# from .q_layer import QuantumLayer
from .attention import Attention

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
        # 闅忔満drop涓€涓畬鏁寸殑block锛?
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

class Transformer_Encoder(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 dim_head_before_qkv_proj: int,
                 mlp_ratio = 4.0,
                 pro_linear_drop = 0.0,
                 pro_attn_drop = 0.0,
                 pro_path_drop = 0.0,
                 qkv_projection: str = 'linear',
                 dim_head_after_qkv_proj: int = None,
                 qkv_mlp_hidden_dim: int = 3,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
        super(Transformer_Encoder, self).__init__()
        self.norm1 = norm_layer(dim_emb)
        self.attn = Attention(dim_emb = dim_emb,
                                num_heads = num_heads,
                                dim_head_before_qkv_proj = dim_head_before_qkv_proj,
                                dim_head_after_qkv_proj = dim_head_after_qkv_proj,
                                pro_attn_drop = pro_attn_drop,
                                qkv_projection = qkv_projection,
                                qkv_mlp_hidden_dim = qkv_mlp_hidden_dim)

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

if __name__ == "__main__":
    # 娴嬭瘯鐢?
    batch_size = 2
    seq_len = 8
    dim_emb = 16
    num_heads = 2
    dim_head_before_qkv_proj = 8

    # 闅忔満杈撳叆
    x = torch.randn(batch_size, seq_len, dim_emb)

    # 鍒濆鍖栫紪鐮佸櫒
    encoder = Transformer_Encoder(
        dim_emb=dim_emb,
        num_heads=num_heads,
        dim_head_before_qkv_proj = dim_head_before_qkv_proj,
        pro_linear_drop=0.1,
        pro_attn_drop=0.1,
        pro_path_drop=0.1
    )

    # 鍓嶅悜浼犳挱
    out = encoder(x)
    print("杈撳叆褰㈢姸:", x.shape)
    print("杈撳嚭褰㈢姸:", out.shape)
    # print(f"q_proj鐨勬墍鏈塳ey:{encoder.attn.q_proj.state_dict()['linear.weights'].shape}")
