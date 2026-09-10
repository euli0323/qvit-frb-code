import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_q.q_model import QViT


def get_model_param_stats(model):
    total_params = 0
    quantum_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if "qc" in name:
            quantum_params += num_params

    classical_params = total_params - quantum_params

    print("\n=========== Parameter Summary ===========")
    print(f"Total parameters: {total_params:,}")
    print(f"Quantum parameters: {quantum_params:,}")
    print(f"Classical parameters: {classical_params:,}")
    if total_params > 0:
        print(f"Quantum parameter ratio: {quantum_params / total_params * 100:.2f}%")

    print("\n=========== Parameter Details ===========")
    for name, param in model.named_parameters():
        print(f"{name:40s} {str(list(param.shape)):20s} {param.numel():10d}")


def analyze_parameters(model, logger):
    if logger is None:
        raise ValueError("logger must be provided")

    module_params = {}
    total_params = 0
    total_trainable = 0
    quantum_params = 0

    for name, param in model.named_parameters():
        module_path = ".".join(name.split(".")[:-1]) or "root"
        module_params.setdefault(module_path, {"trainable": 0, "non_trainable": 0})

        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            module_params[module_path]["trainable"] += num_params
            total_trainable += num_params
        else:
            module_params[module_path]["non_trainable"] += num_params

        if "qc" in name:
            quantum_params += num_params

    classical_params = total_params - quantum_params
    sorted_modules = sorted(
        module_params.items(),
        key=lambda item: item[1]["trainable"] + item[1]["non_trainable"],
        reverse=True,
    )

    logger.info("-" * 120)
    logger.info("Model parameter analysis")
    logger.info("-" * 120)
    for module_path, counts in sorted_modules:
        module_total = counts["trainable"] + counts["non_trainable"]
        if module_total == 0:
            continue
        percent = module_total / total_params * 100 if total_params else 0.0
        logger.info(
            f"{module_path:<50} total: {module_total:>12,} "
            f"trainable: {counts['trainable']:>10,} "
            f"frozen: {counts['non_trainable']:>10,} "
            f"ratio: {percent:>6.2f}%"
        )

    logger.info("-" * 120)
    logger.info(
        f"{'Total':<50} total: {total_params:>12,} "
        f"trainable: {total_trainable:>10,} "
        f"frozen: {total_params - total_trainable:>10,}"
    )
    logger.info(f"Quantum parameters: {quantum_params:,}")
    logger.info(f"Classical parameters: {classical_params:,}")
    if total_params > 0:
        logger.info(f"Quantum parameter ratio: {quantum_params / total_params * 100:.2f}%")
    logger.info("-" * 120)


if __name__ == "__main__":
    model = QViT(
        size_image=(144, 144),
        size_windows=36,
        num_channels=1,
        num_classes=2,
        dim_emb=8,
        index_qubit_end=0,
        num_qlayers=0,
        type_qc="zy",
        depth=1,
        num_heads=4,
        mlp_ratio=3.0,
        pro_attn_drop=0.1,
        pro_linear_drop=0.1,
        method_encoding="angle",
    )
    get_model_param_stats(model)
