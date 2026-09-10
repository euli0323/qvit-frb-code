from typing import Tuple, Union

import torch
import torch.nn as nn

from .position_emb import Position_Embedding
from .patch_emb import Patch_Embedding, ConvStemEmbedding
from .transformer import Transformer_Encoder

class ViT(nn.Module):
    def __init__(self,
                 size_image: Union[int, Tuple[int, int]] = (144, 144),
                 size_windows: Union[int, Tuple[int, int]] = 36,
                 num_channels: int = 1,
                 num_classes: int = 2,
                 dim_emb: int = 512,
                 depth: int = 6,
                 num_heads: int = 4,
                 dim_head_before_qkv_proj: int = 32,
                 mlp_ratio: float = 3.0,
                 pro_attn_drop: float = 0.1,
                 pro_linear_drop: float = 0.1,
                 pro_path_drop: float = 0.0,
                 pos_mode: str = 'learnable',
                 patch_embed: str = 'proj',
                 stem_channels: int = 32,
                 stem_layers: int = 2,
                 stem_dilation: int = 1,
                 qkv_projection: str = 'linear',
                 dim_head_after_qkv_proj: int = None,
                 qkv_mlp_hidden_dim: int = 3):
        super().__init__()
        
        if isinstance(size_image, int):
            h_img = w_img = size_image
        else:
            h_img, w_img = size_image

        if isinstance(size_windows, int):
            h_windows = w_windows = size_windows
        else:
            h_windows, w_windows = size_windows

        
        assert h_img % h_windows == 0 and w_img % w_windows == 0, "image size must be divisible by window size"

        num_patches = (h_img // h_windows) * (w_img // w_windows)

        patch_embed_cls = ConvStemEmbedding if patch_embed == 'conv_stem' else Patch_Embedding
        patch_embed_kwargs = {'stem_channels': stem_channels, 'stem_layers': stem_layers, 'stem_dilation': stem_dilation} if patch_embed == 'conv_stem' else {}
        self.patch_embed = patch_embed_cls(size_image = size_image,
                                          size_windows = size_windows,
                                          num_channels = num_channels,
                                          dim_emb = dim_emb,
                                          **patch_embed_kwargs)
        
        self.pos_embed = Position_Embedding(d_model = dim_emb, 
                                            n_pos = num_patches, 
                                            mode = pos_mode)
        
        self.encoders = nn.ModuleList([Transformer_Encoder
                                 (dim_emb = dim_emb, 
                                  num_heads = num_heads,
                                  dim_head_before_qkv_proj = dim_head_before_qkv_proj,
                                  mlp_ratio = mlp_ratio,
                                  pro_linear_drop = pro_linear_drop, 
                                  pro_attn_drop = pro_attn_drop,
                                  qkv_projection = qkv_projection,
                                  dim_head_after_qkv_proj = dim_head_after_qkv_proj,
                                  qkv_mlp_hidden_dim = qkv_mlp_hidden_dim)
                                  for _ in range(depth)])
        self.norm = nn.LayerNorm(dim_emb)
        self.layer_dim_post_process = nn.Linear(dim_emb, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x, H, W = self.patch_embed(x)   
        x = self.patch_embed(x)          # (B, N, D)

        x = self.pos_embed(x)           # (B, N+1, D)

        for encoder in self.encoders:
            x = encoder(x)             # (B, N+1, D)

        cls = self.norm(x[:, 0])        # (B, D)

        out = self.layer_dim_post_process(cls)  # (B, num_classes)

        return out

if __name__ == "__main__":
    size_image = (680, 850)
    size_batch = 10
    num_channels = 1
    
    model = ViT(
        size_image = size_image,  # H, W
        size_windows = 34,          # 涔熷彲鍐欐垚 (17, 17)
        num_channels = 1,
        num_classes = 2,
        dim_emb = 48,
        depth = 6,
        num_heads = 8,
        dim_head_before_qkv_proj = 6,
        mlp_ratio = 4.0,
        pro_attn_drop=0.1,
        pro_linear_drop = 0.1,
        pro_path_drop = 0.,
        pos_mode = 'learnable')
    
    h, w = size_image
    x = torch.randn(size_batch, num_channels, h, w)
    y = model(x)

    print(f"杈撳叆鐨勫舰鐘?{x.shape}")
    print(f"杈撳嚭鐨勫舰鐘?{y.shape}")  # (size_batch, 2)
