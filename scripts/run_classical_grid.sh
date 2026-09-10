#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../classical_vit"

USE_TEACHER="${USE_TEACHER:-0}"
TEACHER_ARG=()
if [[ "$USE_TEACHER" == "1" ]]; then
  TEACHER_ARG=(--use-teacher true)
else
  TEACHER_ARG=(--use-teacher false)
fi

for before in 3 4 5; do
  for after in 2 4 6 8 10; do
    python scripts/train.py \
      --config configs/default.json \
      "${TEACHER_ARG[@]}" \
      --output-root "outputs/classical_linear_qkv" \
      --set \
        qkv_projection linear \
        dim_head_before_qkv_proj "$before" \
        dim_head_after_qkv_proj "$after" \
        num_heads 4
  done
done
