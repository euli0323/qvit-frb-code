# Manuscript configurations

This note records the model configurations used for the main manuscript comparison.

## Shared setup

- Image size: `288 x 288`
- Patch window: `48 x 48`
- Patch grid: `6 x 6`
- Encoder depth: `1`
- Attention heads: `4`
- Loss: focal loss, `gamma = 2.0`
- Optimizer: AdamW
- Learning rate: `4e-4`
- Weight decay: `1e-2`
- Warm-up epochs: `8`
- Maximum epochs: `120`
- Early stopping patience: `40`
- Independent seeds per configuration: `10`
- Primary model-selection metric: validation-set FRB-class `F_2`

## Classical ViT

Selected configuration:

```text
linear_emb20_inner24
```

This corresponds to:

```text
dim_head_before_qkv_proj = 5
dim_head_after_qkv_proj  = 6
num_heads                = 4
dim_emb                  = 20
```

Training command:

```bash
cd classical_vit
python scripts/train.py --config configs/default.json \
  --use-teacher false \
  --set \
    qkv_projection linear \
    dim_head_before_qkv_proj 5 \
    dim_head_after_qkv_proj 6 \
    num_heads 4
```

## QViT

Selected configuration:

```text
q4_circuit_depth1_end4
```

This corresponds to:

```text
num_qubits       = 4
circuit_depth    = 1
index_qubit_end  = 4
num_heads        = 4
dim_emb          = 16
```

Training command:

```bash
cd quantum_qvit
python scripts/train.py --config configs/default.json \
  --use-teacher false \
  --num-qubits 4 \
  --circuit-depth 1
```

## Evaluation

Full test-set evaluation:

```bash
cd classical_vit
python scripts/res_test.py --data-dir data/yxdata/test --res-train-dirs outputs/classical_linear_qkv

cd ../quantum_qvit
python scripts/res_test.py --data-dir data/yxdata/test --res-train-dirs outputs/qvit
```

SNR-stratified evaluation:

```bash
cd classical_vit
python scripts/res_test_snr.py --data-dir data/yxdata/test --res-train-dirs outputs/classical_linear_qkv

cd ../quantum_qvit
python scripts/res_test_snr.py --data-dir data/yxdata/test --res-train-dirs outputs/qvit
```
