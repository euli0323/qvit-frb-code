"""从测量 counts 计算每比特 Z 期望值。"""

from typing import Dict, List, Literal


def exp_from_count(
    count: Dict[str, int],
    num_qubits: int,
    bitstring_left: Literal["LSB", "MSB"] = "LSB",
) -> List[float]:
    """
    从 counts 计算 Z 期望值列表。

    LSB: bitstring 左→右 = q0→q_{n-1}（与 generate_qasm / 天眼约定一致）
    MSB: bitstring 左→右 = q_{n-1}→q0
    """
    total = sum(count.values())
    if total == 0:
        return [0.0] * num_qubits

    exp = [0.0] * num_qubits
    for bitstring, c in count.items():
        p = c / total
        if bitstring_left == "LSB":
            bits = bitstring
        else:
            bits = bitstring[::-1]
        for i in range(num_qubits):
            if i < len(bits):
                exp[i] += p * (1.0 if bits[i] == "0" else -1.0)
    return exp
