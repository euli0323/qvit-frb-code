#!/usr/bin/env bash
set -euo pipefail

USE_TEACHER="${USE_TEACHER:-0}"
if [[ "$USE_TEACHER" == "1" || "$USE_TEACHER" == "true" ]]; then
  TEACHER_FLAG="true"
else
  TEACHER_FLAG="false"
fi

for num_qubits in 3 4 5; do
  for circuit_depth in 0 1 2 3 4; do
    python scripts/train.py --config configs/default.json \
      --use-teacher "$TEACHER_FLAG" \
      --num-qubits "$num_qubits" \
      --circuit-depth "$circuit_depth"
  done
done
