#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="D:/github/dataset/imagenet"
if [ ! -d "${DATA_ROOT}" ]; then
  echo "Set DATA_ROOT to a valid ImageNet path before running this script."
  exit 1
fi

cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/train.py" \
  --task classification \
  --dataset imagenet \
  --data-dir "${DATA_ROOT}" \
  --arch resnet50 \
  --num-classes 1000 \
  --image-size 224 \
  --epochs 120 \
  --batch-size 256 \
  --workers 8 \
  --lr 0.1 \
  --bits 2,3,4,5,6 \
  --target-bitops 65.0 \
  --output-dir runs/imagenet_resnet50
