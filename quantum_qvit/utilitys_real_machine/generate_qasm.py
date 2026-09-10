import torch

# util: safe scalar
def _p(x):
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)

def add_zyz_layer(gates, weights_layer, num_qubits):
    """
    每个 qubit: Z-Y-Z (3 params)
    weights_layer shape: [3 * num_qubits]
    """

    for q in range(num_qubits):
        idx = 3 * q

        gates.append({
            "gate": "rz",
            "target": q,
            "params": [_p(weights_layer[idx])]
        })

        gates.append({
            "gate": "ry",
            "target": q,
            "params": [_p(weights_layer[idx + 1])]
        })

        gates.append({
            "gate": "rz",
            "target": q,
            "params": [_p(weights_layer[idx + 2])]
        })

def add_zy_layer(gates, weights_layer, num_qubits):
    for q in range(num_qubits):
        idx = 2 * q

        gates.append({
            "gate": "rz",
            "target": q,
            "params": [_p(weights_layer[idx])]
        })

        gates.append({
            "gate": "ry",
            "target": q,
            "params": [_p(weights_layer[idx + 1])]
        })

def add_z_layer(gates, weights_layer, num_qubits):

    for q in range(num_qubits):
        idx = q

        gates.append({
            "gate": "rz",
            "target": q,
            "params": [_p(weights_layer[idx])]
        })

def add_param_layer(gates, weights_layer, num_qubits, type_qc, index_qubit_end=None):
    if type_qc == "z":
        add_z_layer(gates, weights_layer, num_qubits)
        return

    if type_qc == "zy":
        end = num_qubits if index_qubit_end is None else index_qubit_end
        if not (0 <= end <= num_qubits):
            raise ValueError(f"index_qubit_end 超出范围: {end}, num_qubits={num_qubits}")
        add_zy_layer(gates, weights_layer, end)
        return

    if type_qc == "zyz":
        add_zyz_layer(gates, weights_layer, num_qubits)
        return

    raise ValueError(f"Unsupported type_qc: {type_qc}")

def add_rx_encoding(gates, x_token, num_qubits):
    for i in range(num_qubits):
        gates.append({
            "gate": "rx",
            "target": i,
            "params": [_p(x_token[i])]
        })

def add_cz_chain(gates, num_qubits):
    for q in range(num_qubits - 1):
        gates.append({
            "gate": "cz",
            "control": q,
            "target": q + 1
        })

def gates_to_qasm(gates, num_qubits):
    lines = []

    lines.append("OPENQASM 2.0;")
    lines.append('include "qelib1.inc";')
    lines.append(f"qreg q[{num_qubits}];")
    lines.append(f"creg meas[{num_qubits}];")

    for g in gates:
        name = g["gate"]

        if name in ["rx", "ry", "rz"]:
            theta = g["params"][0]
            lines.append(f"{name}({theta:.6f}) q[{g['target']}];")

        elif name == "cz":
            lines.append(f"cz q[{g['control']}],q[{g['target']}];")

        else:
            raise ValueError(f"Unsupported gate: {name}")

    for i in range(num_qubits):
        lines.append(f"measure q[{i}] -> meas[{i}];")

    res = "\n".join(lines)
    
    return res

# =========================
# 5. main generator (核心)
# =========================
def build_gates(
    num_qubits: int,
    x_token: torch.Tensor,
    weights: torch.Tensor,
    type_qc: str = "zy",
    index_qubit_end: int = None,
):
    """
    固定结构门序列 generator（不转 QASM 字符串，保留完整浮点精度）：

    weights:
        [0]       → initial ZYZ
        [1..L]    → variational ZYZ
    """

    x_token = x_token.detach().cpu()
    weights = weights.detach().cpu()

    gates = []

    # ===== check =====
    if len(x_token) != num_qubits:
        raise ValueError(f"x_token size mismatch: {len(x_token)} vs {num_qubits}")

    num_params = len(type_qc)
    expected_width = num_params * num_qubits
    if weights.shape[1] != expected_width:
        raise ValueError(
            f"weights shape error: got {weights.shape[1]}, expected {expected_width} "
            f"(type_qc={type_qc}, num_qubits={num_qubits})"
        )

    '增强层'
    add_param_layer(gates, weights[0], num_qubits, type_qc, index_qubit_end)
    add_cz_chain(gates, num_qubits)

    '编码层'
    add_rx_encoding(gates, x_token, num_qubits)

    num_layers = weights.shape[0] - 1

    '主层循环'
    for l in range(1, num_layers + 1):
        add_param_layer(gates, weights[l], num_qubits, type_qc, index_qubit_end)
        add_cz_chain(gates, num_qubits)

    return gates


def generate_qasm(
    num_qubits: int,
    x_token: torch.Tensor,
    weights: torch.Tensor,
    type_qc: str = "zy",
    index_qubit_end: int = None,
):
    """
    固定结构 QASM generator：

    weights:
        [0]       → initial ZYZ
        [1..L]    → variational ZYZ
    """

    gates = build_gates(num_qubits, x_token, weights, type_qc, index_qubit_end)

    # ===== compile =====
    return gates_to_qasm(gates, num_qubits)

if __name__ == "__main__":
    import torch

    # 1. 基本参数
    num_qubits = 3
    num_layers = 2

    x_token = torch.randn(num_qubits)

    weights = torch.randn(num_layers + 1, 2*num_qubits)

    # 2. 调用生成 QASM
    qasm = generate_qasm(
        num_qubits=num_qubits,
        x_token=x_token,
        weights=weights
    )

    # 5. 输出
    print("\n================QASM输出================\n")
    print(qasm)


    # 6. 简单结构检查
    print("检查点：")

    assert "rz(" in qasm, "缺少 RZ 门"
    assert "ry(" in qasm, "缺少 RY 门"
    assert "rx(" in qasm, "缺少 RX 编码"
    assert "cz" in qasm, "缺少 CZ 纠缠"

    print("✔ 门类型检查通过")

    # qubit 数检查
    assert qasm.count("q[0]") > 0
    assert qasm.count("q[1]") > 0
    assert qasm.count("q[2]") > 0

    print("✔ qubit 使用检查通过")

    print("\n🎉 测试全部通过！")