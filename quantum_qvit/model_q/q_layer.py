import sys, logging, time, json, dataclasses

from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# logger = logging.getLogger(__name__)

'官方库'
from typing import Literal
import torch
import pennylane as qml

'将项目根目录加入Python搜索路径'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

'自写库'
from utilitys_real_machine.run_on_real import run_real_machine
from utilitys_real_machine.generate_qasm import generate_qasm, build_gates


def _is_tianyan_device(name) -> bool:
    '判断是否为天眼(tianyan)系真机'
    return name is not None and str(name).lower().startswith("tianyan")

'量子神经网络层'
class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits: int,
        index_qubit_end:int,
        num_qlayers: int,
        type_qc: Literal['z', 'zy', 'zyz'] = 'z',
        method_encoding: Literal["angle", "amplitude"] = "angle",
        qdevice: str = "default.qubit",
        diff_method: str = "best",
        ansatz: Literal["hardware_efficient", "enhanced_hardware_efficient"] = "enhanced_hardware_efficient",
        use_real: bool = False,
        api: str = None,
        device_real: Literal['Baihua', 'Yudu', 'Dongling', 'Hongluo', 'Baiwang']|None = None,
        type_qkv: Literal['q', 'k', 'v']|None = None,
        id_encoder: int|None = None,
        need_save_res: bool = False,
        path_base: Path = None,
        logger: logging.Logger = None,
        real_qc_recorder = None,
        sim_qc_recorder = None,
        wait_timeout_seconds: float | None = 300,
        ):
        super().__init__()
        '日志'
        self.logger = logger
        self.real_qc_recorder = real_qc_recorder
        self.sim_qc_recorder = sim_qc_recorder
        self.wait_timeout_seconds = wait_timeout_seconds

        '保存超参数'
        self.num_qubits = num_qubits
        self.index_qubit_end = self.num_qubits if index_qubit_end is None else index_qubit_end

        if not (0 <= self.index_qubit_end <= self.num_qubits):
            raise ValueError(f"index_qubit_end 超出范围")

        self.num_qlayers = num_qlayers
        self.type_qc = type_qc
        self.method_encoding = method_encoding
        self.ansatz = ansatz

        '记录模型的id信息'
        self.type_qkv = type_qkv
        self.id_encoder = id_encoder

        '保存结果的路径'
        self.need_save_res = need_save_res

        '真机参数'
        self.use_real = use_real
        self.api = api
        self.device_real = device_real
        
        '判断是否使用真机推理'
        if self.use_real:

            '判断是否传入了api'
            if not self.api:
                raise ValueError(f"调用真机时api不能为空")

            '判断是否传入了设备名称'
            if not self.device_real:
                raise ValueError(f"调用真机推理时设备名不能为空")

        '创建量子设备'
        dev = qml.device(qdevice, wires=num_qubits)

        '根据线路类型,判断参数量'
        self.num_per_qubit_per_layer = len(type_qc)

        '判断使用哪个block'
        if self.num_per_qubit_per_layer == 1:
            self._block_fn = self._block1

        elif self.num_per_qubit_per_layer == 2:
            self._block_fn = self._block2

        elif self.num_per_qubit_per_layer == 3:
            self._block_fn = self._block3

        else:
            raise ValueError(
                f"不支持的 num_per_qubit_per_layer: {self.num_per_qubit_per_layer}"
            )
        '选择电路'
        if ansatz == "hardware_efficient":
            circuit_fn = self._circuit_hardware_efficient
            weights_shape = {"weights": (num_qlayers, self.num_per_qubit_per_layer * num_qubits)}

        elif ansatz == "enhanced_hardware_efficient":
            circuit_fn = self._circuit_enhanced
            weights_shape = {"weights": (num_qlayers + 1, self.num_per_qubit_per_layer * num_qubits)}
            
        else:
            raise ValueError(f"未知 ansatz: {ansatz}")

        '创建 QNode'
        qnode = qml.QNode(
            circuit_fn,
            dev,
            interface = "torch",
            diff_method = diff_method
        )

        '封装为 TorchLayer'
        self.qc = qml.qnn.TorchLayer(qnode, weights_shape)

    '编码函数（统一管理）'
    def _encode(self, inputs):
        if self.method_encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))

        elif self.method_encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)

        else:
            raise ValueError(f"编码方式错误：{self.method_encoding}")

    '便分层1'
    def _block1(self, weights, index_layer):
        for qubit in range(self.num_qubits):
            idx = qubit
            qml.RZ(weights[index_layer, idx], wires=qubit)

        for qubit in range(self.num_qubits - 1):
            qml.CZ(wires=[qubit, qubit + 1])

    '变分层2'
    def _block2(self, weights, index_layer):
        
        for qubit in range(self.index_qubit_end):
            idx = 2 * qubit
            qml.RZ(weights[index_layer, idx], wires=qubit)
            qml.RY(weights[index_layer, idx + 1], wires=qubit)

        for qubit in range(self.num_qubits - 1):
            qml.CZ(wires=[qubit, qubit + 1])

    '变分层3'
    def _block3(self, weights, index_layer):
        for qubit in range(self.num_qubits):
            idx = 3 * qubit
            qml.RZ(weights[index_layer, idx], wires=qubit)
            qml.RY(weights[index_layer, idx + 1], wires=qubit)
            qml.RZ(weights[index_layer, idx + 2], wires=qubit)

        for qubit in range(self.num_qubits - 1):
            qml.CZ(wires=[qubit, qubit + 1])

    '标准线路'
    def _circuit_hardware_efficient(self, inputs, weights):

        '编码'
        self._encode(inputs)

        'ansatz'
        for layer in range(self.num_qlayers):
            self._block_fn(weights, layer)

        '测量'
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    '增强线路'
    def _circuit_enhanced(self, inputs, weights):

        '第一层（无编码）'
        self._block_fn(weights, 0)

        '中间插入 embedding'
        self._encode(inputs)

        '后续层'
        for layer in range(1, self.num_qlayers + 1):
            self._block_fn(weights, layer)

        '测量'
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _inference_real_machine(self, x):

        '保存原始数据的信息'
        original_device = x.device
        original_dtype = x.dtype

        '形状'
        B, heads, N, dim_head = x.shape

        '权重'
        weights = self.qc.state_dict()['weights']

        'x展平'
        x_flat = x.reshape(-1, dim_head)
        total = x_flat.shape[0]

        '数据移动至cpu'
        x_flat_cpu = x_flat.cpu()
        weights_cpu = weights.cpu()

        '创建存放结果的list'
        outputs = []

        'tianyan 系真机直接由 tensor 构建门序列, 北量院维持 qasm'
        is_tianyan = _is_tianyan_device(self.device_real)

        '遍历每个token'
        for i in range(total):
            x_token = x_flat_cpu[i]
            batch_index = i // (heads * N)
            rem = i % (heads * N)
            head_index = rem // N
            token_index = rem % N

            metadata = {
                "encoder_id": self.id_encoder,
                "qkv": self.type_qkv,
                "batch_index": batch_index,
                "head_index": head_index,
                "token_index": token_index,
                "flat_index": i,
                "num_heads": heads,
                "num_tokens": N,
                "num_qubits": self.num_qubits,
            }

            '创建线路: tianyan 直接构建门序列, 北量院生成 qasm 字符串'
            if is_tianyan:
                circuit_repr = build_gates(
                    self.num_qubits,
                    x_token,
                    weights_cpu,
                    type_qc=self.type_qc,
                    index_qubit_end=self.index_qubit_end,
                )
            else:
                circuit_repr = generate_qasm(
                    self.num_qubits,
                    x_token,
                    weights_cpu,
                    type_qc=self.type_qc,
                    index_qubit_end=self.index_qubit_end,
                )
            id_circuit = i + 1

            if self.real_qc_recorder is not None:
                cached = self.real_qc_recorder.get_cached_circuit(metadata)
                if cached is not None:
                    exp = torch.tensor(cached["exp"], dtype=torch.float32)
                    outputs.append(exp)
                    if self.logger is not None:
                        self.logger.info(
                            f"encoder{self.id_encoder}|{self.type_qkv}|"
                            f"head{head_index}|token{token_index}|"
                            f"任务{id_circuit}/{total}已存在, 跳过真机调用"
                        )
                    continue

                name_task = self.real_qc_recorder.make_task_name(metadata)
                self.real_qc_recorder.mark_circuit_started(metadata, circuit_repr)
            else:
                name_task = f"encoder{self.id_encoder}_{self.type_qkv}_{id_circuit}"

            '运行真机'
            if is_tianyan:
                res = run_real_machine(gates = circuit_repr,
                                       api = self.api,
                                       chip = self.device_real,
                                       task_name = name_task,
                                       shots = 20000,
                                       timeout = self.wait_timeout_seconds)
            else:
                res = run_real_machine(qasm = circuit_repr,
                                       api = self.api,
                                       chip = self.device_real,
                                       task_name = name_task,
                                       shots = 20000,
                                       timeout = self.wait_timeout_seconds)

            exp = res.get_exp_res(self.num_qubits)
            outputs.append(exp)
            if self.logger is not None:
                self.logger.info(
                    f"encoder{self.id_encoder}|{self.type_qkv}|"
                    f"head{head_index}|token{token_index}|任务{id_circuit}/{total}已完成"
                )

            if self.real_qc_recorder is not None:
                self.real_qc_recorder.record_circuit(
                    metadata=metadata,
                    qasm=circuit_repr,
                    task_name=name_task,
                    result=dataclasses.asdict(res),
                    exp=exp.cpu().tolist(),
                )

        '堆叠, (B*heads*N, dim_emb)'
        y_flat_cpu = torch.stack(outputs)

        '移动回gpu,保持数据类型一致'
        y_flat_gpu = y_flat_cpu.to(original_device, dtype = original_dtype)

        '恢复形状'
        y = y_flat_gpu.reshape(B, heads, N, dim_head)

        return y

    def _inference_simulator_with_recorder(self, x):
        original_device = x.device
        original_dtype = x.dtype

        B, heads, N, dim_head = x.shape
        x_flat = x.reshape(-1, dim_head)
        y_flat = self.qc(x_flat)
        total = y_flat.shape[0]

        for i in range(total):
            batch_index = i // (heads * N)
            rem = i % (heads * N)
            head_index = rem // N
            token_index = rem % N
            metadata = {
                "encoder_id": self.id_encoder,
                "qkv": self.type_qkv,
                "batch_index": batch_index,
                "head_index": head_index,
                "token_index": token_index,
                "flat_index": i,
                "num_heads": heads,
                "num_tokens": N,
                "num_qubits": self.num_qubits,
            }

            if self.sim_qc_recorder is not None:
                self.sim_qc_recorder.mark_circuit_started(metadata)
                exp = y_flat[i].detach().cpu().to(torch.float32).tolist()
                self.sim_qc_recorder.record_circuit(metadata=metadata, exp=exp)

        y = y_flat.reshape(B, heads, N, dim_head).to(original_device, dtype=original_dtype)
        return y

    def forward(self, x):
        '推理 + 真机'
        if not self.training and self.use_real:
            return self._inference_real_machine(x)

        if not self.training and (self.sim_qc_recorder is not None):
            return self._inference_simulator_with_recorder(x)

        return self.qc(x)

if __name__ == "__main__":
    'QuantumLayer 测试开始'
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')

    batch = 1
    N = 5
    num_heads = 2
    num_qubits = 3

    x1 = torch.randn(batch, num_heads, N, num_qubits)  # 输入维度与量子比特数一致
    x2 = torch.randn(2, 8, 10, 32)
    
    'api'
    api = 'V7MvvZU:DVxdDyUx7rEDrzsQnSTcHJHi2PsGUpW4zqi/14O4hENxhENvZEP5l{O4d{O4FkPjBIfmKDMjZEO7REO7FUNhNENuRENuZkNxJkJ7JDeimnJtBkPjxX[3WHcjxjJu:3ZvNkOyBVbtWY[gSndiincwWHcjpkJzW3d2Kzf'

    '实例化量子层'
    qlayer1 = QuantumLayer(
        num_qubits = num_qubits,
        num_qlayers = 1,
        method_encoding="angle",
        use_real=True,
        api = api,
        device_real = 'Hongluo'
    )
    print(f"内部权重的形状:\n{qlayer1.qc.state_dict()['weights'].shape}\n")

    # qlayer2 = QuantumLayer(
    #     num_qubits=5,
    #     num_qlayers=2,
    #     method_encoding="amplitude"
    # )
    qlayer1.eval()

    '前向传播'
    with torch.no_grad():
        y1 = qlayer1(x1)
    # y2 = qlayer2(x2)

    print(f"x1的形状:\n{x1.shape}\n")
    # print(f"输出y1:\n{y1}\n")
    print(f"y1形状:\n{y1.shape}\n")

    # print(f"x2的形状:\n{x2.shape}\n")
    # print(f"输出y2:\n{y2}\n")
    # print(f"y2的形状:\n{y2.shape}\n")