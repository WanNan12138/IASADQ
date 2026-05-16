#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="D:/github/dataset/cifar-10-python"
if [ ! -d "${DATA_ROOT}" ]; then
  echo "Set DATA_ROOT to a valid CIFAR-10 path before running this script."
  exit 1
fi

cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/train.py" \
  --task classification \
  --dataset cifar10 \
  --data-dir "${DATA_ROOT}" \
  --arch resnet18 \
  --num-classes 10 \
  --image-size 32 \
  --epochs 200 \
  --batch-size 128 \
  --workers 4 \
  --lr 0.1 \
  --bits 2,3,4,5,6 \
  --output-dir runs/cifar10_resnet18
