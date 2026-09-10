import torch
from torch import nn


def _factorize(n):
    """把整数 n 分解为升序因子列表（每个 >=2），如 170 -> [2, 5, 17]"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n and n > 1:
            factors.append(n)
            break
    return factors


class ConvStemEmbedding(nn.Module):
    """CNN stem 版 patch embedding：用逐级 stride 卷积承担全部下采样，
    替代对原图的直接 resize，层级式提取局部特征以保留细节。
    输出 token 数与 Patch_Embedding 完全一致（grid_h * grid_w），
    transformer 侧超参不受影响。"""

    def __init__(
        self,
        size_image=(680, 850),
        size_windows=34,
        num_channels=1,
        dim_emb=256,
        stem_channels=32,
        stem_layers=None,
        stem_dilation=1,
    ):
        super().__init__()

        if isinstance(size_image, int):
            size_image = (size_image, size_image)
        if isinstance(size_windows, int):
            size_windows = (size_windows, size_windows)

        self.size_image = size_image
        self.size_windows = size_windows

        grid_h = size_image[0] // size_windows[0]
        grid_w = size_image[1] // size_windows[1]
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        if size_image[0] % size_windows[0] != 0 or size_image[1] % size_windows[1] != 0:
            raise ValueError("size_image must be divisible by size_windows")

        fh = _factorize(size_image[0] // grid_h)
        fw = _factorize(size_image[1] // grid_w)

        stages = []
        i = j = 0
        while i < len(fh) or j < len(fw):
            sh = fh[i] if i < len(fh) else 1
            sw = fw[j] if j < len(fw) else 1
            stages.append((sh, sw))
            i += 1
            j += 1

        n_pre = max(0, (stem_layers - len(stages))) if stem_layers else 0

        ops = []
        c_in = num_channels
        for _ in range(n_pre):
            ops.extend([
                nn.Conv2d(c_in, stem_channels, kernel_size=3,
                          padding=stem_dilation, dilation=stem_dilation),
                nn.GELU(),
            ])
            c_in = stem_channels

        for idx, (sh, sw) in enumerate(stages):
            c_out = stem_channels if idx == 0 else min(c_in * 2, 128)
            ops.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, kernel_size=(sh, sw), stride=(sh, sw)),
                nn.GELU(),
            ])
            c_in = c_out
        self.stem = nn.Sequential(*ops) if ops else nn.Identity()

        if c_in != dim_emb:
            self.proj = nn.Conv2d(c_in, dim_emb, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        if (H, W) != self.size_image:
            raise ValueError(
                f"input image size ({H}, {W}) does not match configured size {self.size_image}"
            )

        x = self.stem(x)
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class Patch_Embedding(nn.Module):
    """将 (B, C, H, W) 图像划分为不重叠 patch，并映射到 dim_emb 维度"""

    def __init__(
        self,
        size_image=(680, 850),
        size_windows=34,
        num_channels=1,
        dim_emb=256,
    ):
        super().__init__()

        if isinstance(size_image, int):
            size_image = (size_image, size_image)
        if isinstance(size_windows, int):
            size_windows = (size_windows, size_windows)

        self.size_image = size_image
        self.size_windows = size_windows

        grid_h = size_image[0] // size_windows[0]
        grid_w = size_image[1] // size_windows[1]
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        self.proj = nn.Conv2d(
            num_channels,
            dim_emb,
            kernel_size=size_windows,
            stride=size_windows
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if (H, W) != self.size_image:
            raise ValueError(
                f"input image size ({H}, {W}) does not match configured size {self.size_image}"
            )

        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


if __name__ == "__main__":
    img = torch.randn(2, 1, 144, 144)

    for cls in (Patch_Embedding, ConvStemEmbedding):
        pe = cls(size_image=(144, 144), size_windows=18,
                 num_channels=1, dim_emb=64)
        out = pe(img)
        print(cls.__name__, "->", tuple(out.shape), "| patches:", pe.num_patches,
              "| params:", sum(p.numel() for p in pe.parameters()))
