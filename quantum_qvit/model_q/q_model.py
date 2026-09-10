'官方库'
from typing import Literal, Tuple, Union, Dict
from pathlib import Path
import logging

'第三方库'
import torch
import torch.nn as nn

'自写依赖'
from .position_emb import Position_Embedding
from .patch_emb import Patch_Embedding, ConvStemEmbedding
from .q_transformer import Q_Transformer_Encoder_QMLP, Q_Transformer_Encoder_MLP

'参数统计工具类'
class ParameterCounter:
    """
    量子-经典混合模型参数统计工具
    
    用于准确统计QViT模型中的量子参数和经典参数数量。
    量子参数指QuantumLayer中的可训练参数，经典参数指其他所有nn.Module参数。
    """
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        统计模型中的量子参数和经典参数数量
        
        Args:
            model: QViT模型实例
            
        Returns:
            Dict包含以下键值对:
            - 'quantum_params': 量子参数总数
            - 'classical_params': 经典参数总数
            - 'total_params': 总参数数量
            - 'quantum_layers': 量子层数量
            
        Example:
            >>> model = QViT(num_qubits=4, num_qlayers=2)
            >>> stats = ParameterCounter.count_parameters(model)
            >>> print(f"量子参数: {stats['quantum_params']}")
            >>> print(f"经典参数: {stats['classical_params']}")
        """
        from .q_layer import QuantumLayer
        
        quantum_params = 0
        classical_params = 0
        quantum_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, QuantumLayer):
                quantum_layers += 1
                'QuantumLayer的参数存储在linear.weights中'
                if hasattr(module, 'linear') and hasattr(module.linear, 'weights'):
                    q_params = module.linear.weights.numel()
                    quantum_params += q_params
                    
        '统计所有nn.Parameter（包括QuantumLayer的）'
        for param in model.parameters():
            '检查参数是否属于QuantumLayer'
            is_quantum = False
            for name, module in model.named_modules():
                if isinstance(module, QuantumLayer):
                    if hasattr(module, 'linear') and hasattr(module.linear, 'weights'):
                        if param is module.linear.weights:
                            is_quantum = True
                            break
            
            if not is_quantum:
                classical_params += param.numel()
        
        total_params = quantum_params + classical_params
        
        return {
            'quantum_params': quantum_params,
            'classical_params': classical_params,
            'total_params': total_params,
            'quantum_layers': quantum_layers
        }
    
    @staticmethod
    def print_parameter_summary(model: nn.Module, verbose: bool = True, file=None) -> None:
        """
        打印模型参数统计摘要
        
        Args:
            model: QViT模型实例
            verbose: 是否打印详细信息
            file: 输出文件对象，默认为sys.stdout
        """
        import sys
        if file is None:
            file = sys.stdout
            
        stats = ParameterCounter.count_parameters(model)
        
        print('=' * 60, file=file)
        print('模型参数统计摘要', file=file)
        print('=' * 60, file=file)
        print(f"量子参数数量:     {stats['quantum_params']:,}", file=file)
        print(f"经典参数数量:     {stats['classical_params']:,}", file=file)
        print(f"总参数数量:       {stats['total_params']:,}", file=file)
        print(f"量子层数量:       {stats['quantum_layers']}", file=file)
        print('-' * 60, file=file)
        
        if stats['total_params'] > 0:
            q_ratio = stats['quantum_params'] / stats['total_params'] * 100
            c_ratio = stats['classical_params'] / stats['total_params'] * 100
            print(f"量子参数占比:     {q_ratio:.2f}%", file=file)
            print(f"经典参数占比:     {c_ratio:.2f}%", file=file)
        
        print('=' * 60, file=file)
        
        if verbose:
            print('\n详细参数列表:', file=file)
            print('-' * 60, file=file)
            for name, param in model.named_parameters():
                print(f"{name:50s} {param.numel():>10,}  {str(tuple(param.shape)):>20}", file=file)
            print('=' * 60, file=file)
        
        '确保输出被刷新'
        file.flush() if hasattr(file, 'flush') else None

'单元测试'
def test_parameter_counter():
    """
    参数统计功能的单元测试
    
    验证:
    1. 量子参数计数准确性
    2. 经典参数计数准确性
    3. 总参数计数准确性
    """
    print("运行参数统计单元测试...")
    
    '测试1: 创建小型QViT模型'
    model = QViT(
        size_image=(64, 64),
        size_windows=16,
        num_channels=1,
        num_classes=2,
        dim_emb=16,
        depth=2,
        num_heads=2,
        method_encoding='angle'
    )
    
    '测试2: 统计参数'
    stats = ParameterCounter.count_parameters(model)
    
    '测试3: 验证计数合理性'
    assert stats['quantum_params'] >= 0, "量子参数数量不能为负"
    assert stats['classical_params'] >= 0, "经典参数数量不能为负"
    assert stats['total_params'] == stats['quantum_params'] + stats['classical_params'], \
        "总参数应等于量子参数加经典参数"
    assert stats['quantum_layers'] > 0, "应至少有一个量子层"
    
    '测试4: 打印摘要'
    ParameterCounter.print_parameter_summary(model, verbose=False)
    
    print("\n✓ 所有单元测试通过！")
    return True

# 'Transformer 编码器堆叠'
# class Block_Transformer(nn.Module):
#     def __init__(self, 
#                  dim_emb: int, 
#                  num_heads: int, 
#                  method_encoding: Literal['angle', 'amplitude'],
#                  num_layers: int, 
#                  mlp_ratio: float = 4.0,
#                  pro_attn_drop: float = 0.0, 
#                  pro_drop: float = 0.0):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             Q_Transformer_Encoder(dim_emb = dim_emb, 
#                                   num_heads = num_heads, 
#                                   method_encoding = method_encoding, 
#                                   mlp_ratio = mlp_ratio,
#                                   pro_linear_drop = pro_drop, 
#                                   pro_attn_drop = pro_attn_drop) for _ in range(num_layers)])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for blk in self.layers:
#             x = blk(x)

#         return x

'QViT 主体（支持非正方形输入）'
class QViT(nn.Module):
    def __init__(self,
                 size_image: Union[int, Tuple[int, int]] = (144, 144),
                 size_windows: Union[int, Tuple[int, int]] = 36,
                 num_channels: int = 1,
                 num_classes: int = 2,
                 dim_emb: int = 12,
                 index_qubit_end: int = None,
                 num_qlayers: int = 2,
                 type_qc: Literal['z', 'zy', 'zyz'] = 'z',
                 depth: int = 1,
                 num_heads: int = 4,
                 mlp_ratio: float = 3.0,
                 pro_attn_drop: float = 0.0,
                 pro_linear_drop: float = 0.0,
                 pro_path_drop: float = 0.0,
                 pos_mode: str = 'learnable',
                 patch_embed: str = 'proj',
                 stem_channels: int = 32,
                 stem_layers: int = 2,
                 stem_dilation: int = 1,
                 method_encoding: str = 'angle',
                 use_real: bool = False,
                 api: str = None,
                 device_real: str = None,
                 need_save_res: bool = False,
                 path_base: Path = None,
                 logger: logging.Logger = None,
                 real_qc_recorder = None,
                 sim_qc_recorder = None,
                 wait_timeout_seconds: float | None = 300):

        super().__init__()
        
        '处理图片输入尺寸'
        if isinstance(size_image, int):
            h_img = w_img = size_image
        else:
            h_img, w_img = size_image

        '处理窗口尺寸'
        if isinstance(size_windows, int):
            h_windows = w_windows = size_windows
        else:
            h_windows, w_windows = size_windows

        assert h_img % h_windows == 0 and w_img % w_windows == 0, "图片的高/宽必须能被窗口的高/宽整除"

        '计算一张图片里的patches(等价于tokens)的数量'
        num_patches = (h_img // h_windows) * (w_img // w_windows)

        patch_embed_cls = ConvStemEmbedding if patch_embed == 'conv_stem' else Patch_Embedding
        patch_embed_kwargs = {
            'stem_channels': stem_channels,
            'stem_layers': stem_layers,
            'stem_dilation': stem_dilation,
        } if patch_embed == 'conv_stem' else {}
        self.patch_embed = patch_embed_cls(size_image = size_image,
                                          size_windows = size_windows,
                                          num_channels = num_channels,
                                          dim_emb = dim_emb,
                                          **patch_embed_kwargs)
        
        '位置编码嵌入'
        self.pos_embed = Position_Embedding(d_model = dim_emb, 
                                            n_pos = num_patches, 
                                            mode = pos_mode)
        
        '构建编码器'
        self.encoders = nn.ModuleList([Q_Transformer_Encoder_MLP
                                 (dim_emb = dim_emb, 
                                  num_heads = num_heads, 
                                  index_qubit_end = index_qubit_end,
                                  num_qlayers = num_qlayers,
                                  type_qc = type_qc,
                                  method_encoding = method_encoding, 
                                  mlp_ratio = mlp_ratio,
                                  pro_linear_drop = pro_linear_drop, 
                                  pro_attn_drop = pro_attn_drop,
                                  use_real = use_real,
                                  api = api,
                                  device_real = device_real,
                                  id_encoder = id,
                                  need_save_res = need_save_res,
                                  path_base = path_base,
                                  logger = logger,
                                  real_qc_recorder = real_qc_recorder,
                                  sim_qc_recorder = sim_qc_recorder,
                                  wait_timeout_seconds = wait_timeout_seconds)
                                  for id in range(depth)])
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
        '1.将每张图片分割为多个patch'
        x = self.patch_embed(x)          # (B, N, D)

        '2.位置编码'
        x = self.pos_embed(x)           # (B, N+1, D)
        
        '3.经过多个encoder'
        for encoder in self.encoders:
            x = encoder(x)             # (B, N+1, D)

        '4.取N+1个tokens里的第一个'
        cls = self.norm(x[:, 0])        # (B, D)

        '5.经过维度后处理层输出分类概率'
        out = self.layer_dim_post_process(cls)  # (B, num_classes)

        return out

'测试'
if __name__ == "__main__":
    '设置参数'
    size_image = (144, 144)
    size_batch = 1
    num_channels = 1
    
    model = QViT(
        size_image = size_image,  # H, W
        size_windows = 36,
        num_channels = 1,
        num_classes = 2,
        dim_emb = 12,
        num_qlayers = 2,
        depth = 1,
        num_heads = 4,
        mlp_ratio = 3.0,
        pro_attn_drop=0.1,
        pro_linear_drop = 0.1,
        pro_path_drop = 0.0,
        pos_mode = 'learnable',
        method_encoding = 'angle')
    
    h, w = size_image
    x = torch.randn(size_batch, num_channels, h, w)

    with torch.no_grad():
        y = model(x)

    print(f"输入的形状:{x.shape}")
    print(f"输出的形状:{y.shape}")  # (批次，类别数)
