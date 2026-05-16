# IASADQ

[![DOI](https://zenodo.org/badge/1238913846.svg)](https://doi.org/10.5281/zenodo.20236367)

Instance-aware Smooth Sharpness Adaptive Quantization.

This project reconstructs the merged quantization idea discussed in the thesis on instance-aware dynamic quantization and smooth loss sharpness.

## Core Ideas

- Instance-aware dynamic bit allocation for per-sample, per-layer quantization.
- Sharpness-aware quantization training with three perturbation paths:
  - `independent`
  - `pre_quant`
  - `post_quant`
- Bit-FLOPs regularization for mixed-precision control.
- KL-guided switching between sharpness-aware cases.

## Supported Datasets

- CIFAR-10 pickle format
- CIFAR-100 pickle format
- ImageNet `train/val` folder format

Dataset paths should be specified by the user through `--data-dir`.

Example dataset layouts:

- `path/to/cifar-10-python`
- `path/to/cifar-100-python`
- `path/to/imagenet`

For ImageNet, the expected folder structure is:

```text
path/to/imagenet/
├── train/
└── val/
```

## Supported Models

- `resnet20`
- `resnet18`
- `resnet34`
- `resnet50`
- `mobilenetv2`

## Bit-Width Settings

- `w2/a2` uses dynamic candidates `1,2,3`
- `w4/a4` uses dynamic candidates `2,3,4,5,6`

The script suffix indicates the target bit-width, not a fixed bit-width for every layer.

## Quick Start

### CIFAR-10 / ResNet20

```bash
bash scripts/run/run_cifar10_resnet20_w4a4.sh
```

```powershell
conda activate saq
.\scripts\run_issaq.ps1 -Dataset cifar10 -Model resnet20 -NumClasses 10 -ImageSize 32 -Qw 4 -Qa 4 -TargetBit 4 -BitsChoice 2,3,4,5,6 -TargetBops 0.61
```

### CIFAR-100 / ResNet20

```bash
bash scripts/run/run_cifar100_resnet20_w4a4.sh
```

### ImageNet / ResNet50

```bash
bash scripts/run/run_imagenet_resnet50_w4a4.sh
```


## Direct Training Example

```bash
python train.py \
  --dataset cifar10 \
  --data-dir path/to/cifar-10-python \
  --network resnet20 \
  --num-classes 10 \
  --image-size 32 \
  --batch-size 128 \
  --n_epochs 200 \
  --lr 0.1 \
  --qw 4 \
  --qa 4 \
  --tar_bit 4 \
  --bits_choice 2,3,4,5,6 \
  --target_bops 0.61 \
  --save_path output/cifar10/resnet20/w4a4
```