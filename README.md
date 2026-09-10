# Hardware-aware QViT and classical ViT for FRB segment classification

This repository contains the code package accompanying the manuscript on a hardware-aware Quantum Vision Transformer (QViT) pipeline for fast radio burst (FRB) segment classification. The code is packaged for paper review and reproduction of the classical ViT and quantum QViT experiments used in the manuscript.

The repository intentionally does not include observational data, trained checkpoints, generated result JSON files, logs, or paper figures. Those artifacts can be regenerated after the dataset is prepared locally.

## Repository layout

```text
qvit-frb-code/
  classical_vit/          Classical constrained ViT baseline
    configs/              JSON configuration files
    model/                ViT model, patch embedding and attention modules
    scripts/              Training and evaluation scripts
    utilitys/             Dataset, loss, optimizer and scheduler helpers
  quantum_qvit/           Hardware-aware QViT model
    configs/              JSON configuration files
    model_q/              QViT model and parameterized quantum circuit layers
    scripts/              Training and evaluation scripts
    utilitys_q/           Dataset, loss, optimizer and scheduler helpers
  scripts/                Top-level launch scripts for grid training/evaluation
  docs/                   Notes for reviewers and users
  requirements.txt        Python dependency list
```

## Data policy

No data are included in this release. Prepare the image dataset using the same class convention as the manuscript:

```text
data/yxdata/
  train/0/*.png
  train/1/*.png
  val/0/*.png
  val/1/*.png
  test/0/*.png
  test/1/*.png
```

The class mapping is fixed:

```text
0 = FRB-like astrophysical segment
1 = RFI or non-astrophysical segment
```

The scripts verify the `ImageFolder` class mapping. The same folder convention should be used for the source-held-out FRB 20220912A evaluation set if that analysis is reproduced.

## Environment

The experiments were run in a local Conda environment named `pytorch`.

```bash
conda activate pytorch
pip install -r requirements.txt
```

The QViT model depends on PennyLane for differentiable quantum-circuit simulation. GPU acceleration is used automatically when CUDA is available.

## Classical ViT baseline

The classical model is located in:

```text
classical_vit/
```

Main files:

- `classical_vit/model/model.py`: ViT classifier.
- `classical_vit/model/attention.py`: constrained linear or MLP Q/K/V projections.
- `classical_vit/scripts/train.py`: repeated-seed training entry point.
- `classical_vit/scripts/res_test.py`: full test-set evaluation.
- `classical_vit/scripts/res_test_snr.py`: SNR-stratified evaluation.

Single classical run:

```bash
cd classical_vit
python scripts/train.py --config configs/default.json \
  --use-teacher false \
  --output-root outputs/classical_linear_qkv \
  --set \
    qkv_projection linear \
    dim_head_before_qkv_proj 5 \
    dim_head_after_qkv_proj 6 \
    num_heads 4
```

Full classical sweep:

```bash
cd /path/to/qvit-frb-code
bash scripts/run_classical_grid.sh
```

The classical sweep varies:

- `dim_head_before_qkv_proj`: pre-projection head dimension, typically `3, 4, 5`.
- `dim_head_after_qkv_proj`: internal Q/K/V projection dimension, typically `2, 4, 6, 8, 10`.
- `num_heads`: fixed to `4` in the manuscript experiments.
- `qkv_projection`: `linear` in the main constrained baseline; `mlp` is supported by the code.

The selected manuscript baseline uses `linear_emb20_inner24`, corresponding to `dim_head_before_qkv_proj=5`, `dim_head_after_qkv_proj=6`, and `num_heads=4`.

## Quantum QViT

The quantum model is located in:

```text
quantum_qvit/
```

Main files:

- `quantum_qvit/model_q/q_model.py`: QViT classifier.
- `quantum_qvit/model_q/q_attention.py`: quantum Q/K/V attention projections.
- `quantum_qvit/model_q/q_layer.py`: parameterized quantum circuit layer.
- `quantum_qvit/scripts/train.py`: repeated-seed training entry point.
- `quantum_qvit/scripts/res_test.py`: full test-set evaluation.
- `quantum_qvit/scripts/res_test_snr.py`: SNR-stratified evaluation.

Single QViT run:

```bash
cd quantum_qvit
python scripts/train.py --config configs/default.json \
  --use-teacher false \
  --output-root outputs/qvit \
  --num-qubits 4 \
  --circuit-depth 1
```

Full QViT sweep:

```bash
cd /path/to/qvit-frb-code
bash scripts/run_quantum_grid.sh
```

The QViT sweep varies:

- `num_qubits`: qubits per attention head, typically `3, 4, 5`.
- `circuit_depth`: number of repeated trainable quantum layers, typically `0, 1, 2, 3, 4`.
- `index_qubit_end`: number of measured/output qubits; by default it follows `num_qubits`.
- `type_qc`: circuit template identifier; the manuscript configuration uses `zy`.
- `method_encoding`: input encoding method; the manuscript configuration uses angle encoding.
- `num_heads`: fixed to `4` in the manuscript experiments.

The selected manuscript QViT uses `q4_circuit_depth1_end4`, corresponding to 4 qubits per head, circuit depth 1, and 4 measured output qubits per head.

## Shared training configuration

Both branches use the same high-level image and optimization setup in the manuscript experiments:

- Input: single-channel dynamic-spectrum image.
- Image size: `288 x 288`.
- Patch window: `48 x 48`, producing a `6 x 6` patch grid plus the classification token.
- Encoder depth: `1`.
- Number of heads: `4`.
- Loss: focal loss with `gamma=2.0`.
- Optimizer: AdamW.
- Learning rate: `4e-4`.
- Weight decay: `1e-2`.
- Warm-up: 8 epochs followed by cosine decay.
- Batch size: 32.
- Maximum epochs: 120.
- Early stopping patience: 40 epochs.
- Seed count: 10 independent runs.
- Main selection metric: validation-set FRB-class `F_2`.

Most settings can be changed in the JSON config files or through command-line overrides:

```bash
python scripts/train.py --config configs/default.json --set batch_size 16 lr 0.0002 seeds 3
```

## Optional teacher distillation

The training scripts support an optional teacher model through:

```bash
--use-teacher true --teacher path/to/best_model.pth
```

No teacher checkpoint is included in this repository. This keeps the release lightweight and avoids distributing trained model weights with the code package. If a teacher is not supplied, use:

```bash
--use-teacher false
```

## Evaluation

After training, run full test-set and SNR-stratified evaluation.

Classical:

```bash
bash scripts/evaluate_classical.sh
```

QViT:

```bash
bash scripts/evaluate_quantum.sh
```

Environment variables can override default paths:

```bash
DATA_DIR=data/yxdata/test TRAIN_DIR=outputs/qvit OUT_DIR=eval_results bash scripts/evaluate_quantum.sh
```

The evaluation JSON files contain per-seed metrics grouped by configuration. The main reported metrics are accuracy, recall, precision and `F_2` for the FRB-like class.

## Reproducing manuscript model choices

The manuscript reports the selected model configurations:

```text
Classical ViT: linear_emb20_inner24
QViT:          q4_circuit_depth1_end4
```

For the classical model, `linear_emb20_inner24` corresponds to an embedding dimension of 20 and an internal Q/K/V projection dimension of 24 across four heads. For QViT, `q4_circuit_depth1_end4` corresponds to four qubits per head, one quantum circuit layer, and four measured output qubits.

## Notes for GitHub release

Before uploading to GitHub, keep the following out of the repository:

- Raw PSRFITS files and generated PNG datasets.
- `data/`, `datasets/`, and any source-specific private data directory.
- `*.pth`, `*.pt`, and `*.ckpt` checkpoint files.
- Training logs and generated result folders such as `res_c*`, `res_q*`, `outputs/`, `eval_results/`.
- Generated manuscript figures unless a journal or archive requires them separately.

The `.gitignore` file in this release directory is configured to exclude those artifacts.
