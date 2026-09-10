#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../classical_vit"

DATA_DIR="${DATA_DIR:-data/yxdata/test}"
TRAIN_DIR="${TRAIN_DIR:-outputs/classical_linear_qkv}"
OUT_DIR="${OUT_DIR:-eval_results}"

python scripts/res_test.py \
  --data-dir "$DATA_DIR" \
  --res-train-dirs "$TRAIN_DIR" \
  --out-dir "$OUT_DIR/res_test"

python scripts/res_test_snr.py \
  --data-dir "$DATA_DIR" \
  --res-train-dirs "$TRAIN_DIR" \
  --out-dir "$OUT_DIR/res_test_snr"
