#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../quantum_qvit"

USE_TEACHER="${USE_TEACHER:-0}"
TEACHER_ARG=()
if [[ "$USE_TEACHER" == "1" ]]; then
  TEACHER_ARG=(--use-teacher true)
else
  TEACHER_ARG=(--use-teacher false)
fi

for qubits in 3 4 5; do
  for depth in 0 1 2 3 4; do
    python scripts/train.py \
      --config configs/default.json \
      "${TEACHER_ARG[@]}" \
      --output-root "outputs/qvit" \
      --num-qubits "$qubits" \
      --circuit-depth "$depth"
  done
done
