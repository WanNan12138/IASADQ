from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def _coerce_yaml_value(value):
    if value in {"", "null", "Null", "NULL", "~"}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        items = []
        for item in value[1:-1].split(","):
            item = item.strip()
            if item:
                items.append(_coerce_yaml_value(item))
        return items
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(path):
    data = {}
    active_key = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith(" ") or line.startswith("\t"):
            if active_key is None:
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                container = data.get(active_key)
                if not isinstance(container, list):
                    container = []
                    data[active_key] = container
                container.append(_coerce_yaml_value(stripped[2:].strip()))
                continue
            if ":" in stripped:
                nested_key, nested_value = stripped.split(":", 1)
                container = data.get(active_key)
                if not isinstance(container, dict):
                    container = {}
                    data[active_key] = container
                container[nested_key.strip()] = _coerce_yaml_value(nested_value.strip())
            continue

        key, sep, value = line.partition(":")
        if not sep:
            continue
        active_key = key.strip()
        value = value.strip()
        if value:
            data[active_key] = _coerce_yaml_value(value)
        else:
            data[active_key] = []
    return data


def _load_data_spec(path):
    spec_path = Path(path)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        data_spec = _parse_simple_yaml(spec_path)
    else:
        data_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data_spec, dict):
        raise ValueError(f"Invalid detection data spec: {spec_path}")
    return data_spec


def _resolve_data_path(data_spec, spec_path, split_name):
    split_value = data_spec.get(split_name, "")
    if isinstance(split_value, (list, tuple)):
        if len(split_value) != 1:
            raise ValueError(f"Detection data spec field '{split_name}' must be a string or single-item list.")
        split_value = split_value[0]
    if not split_value:
        raise ValueError(f"Detection data spec is missing '{split_name}'.")

    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path

    base_root = data_spec.get("path", "")
    if base_root:
        base_path = Path(str(base_root))
        if not base_path.is_absolute():
            base_path = spec_path.parent / base_path
        return (base_path / split_path).resolve()

    return (spec_path.parent / split_path).resolve()


def _xywhn_to_xyxy(boxes, size):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = (cx - w / 2.0) * size
    y1 = (cy - h / 2.0) * size
    x2 = (cx + w / 2.0) * size
    y2 = (cy + h / 2.0) * size
    return torch.stack([x1, y1, x2, y2], dim=1)


class YoloDetectionDataset(Dataset):
    def __init__(self, image_dir, image_size=640):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"YOLO image directory does not exist: {self.image_dir}")
        self.image_files = sorted(
            [
                path
                for path in self.image_dir.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        label_root = str(self.image_dir).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
        self.label_dir = Path(label_root)

    def __len__(self):
        return len(self.image_files)

    def _read_labels(self, image_path):
        label_path = self.label_dir / image_path.relative_to(self.image_dir).with_suffix(".txt")
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = [float(v) for v in line.strip().split()]
            if len(parts) == 5:
                rows.append(parts)
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image = TF.to_tensor(image)
        labels = self._read_labels(image_path)
        target = {
            "labels": labels[:, 0].long() if labels.numel() else torch.zeros((0,), dtype=torch.long),
            "boxes": _xywhn_to_xyxy(labels[:, 1:], self.image_size) if labels.numel() else torch.zeros((0, 4)),
            "raw": labels,
            "image_id": torch.tensor([index], dtype=torch.long),
        }
        return image, target


def yolo_collate(batch):
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = []
    merged = []
    for batch_index, (_, target) in enumerate(batch):
        targets.append(target)
        if target["raw"].numel():
            batch_column = torch.full((target["raw"].shape[0], 1), batch_index, dtype=target["raw"].dtype)
            merged.append(torch.cat([batch_column, target["raw"]], dim=1))
    merged_targets = torch.cat(merged, dim=0) if merged else torch.zeros((0, 6), dtype=torch.float32)
    return images, {"targets": targets, "yolo_targets": merged_targets}


def build_detection_loaders(args):
    if not args.det_data:
        raise ValueError("Detection task requires --det-data pointing to a YOLO data yaml file.")
    spec_path = Path(args.det_data).resolve()
    data_spec = _load_data_spec(spec_path)
    train_root = _resolve_data_path(data_spec, spec_path, "train")
    val_root = _resolve_data_path(data_spec, spec_path, "val")
    train_set = YoloDetectionDataset(train_root, image_size=args.image_size)
    val_set = YoloDetectionDataset(val_root, image_size=args.image_size)
    if len(train_set) == 0:
        raise FileNotFoundError(f"No images found under YOLO train directory: {train_root}")
    if len(val_set) == 0:
        raise FileNotFoundError(f"No images found under YOLO val directory: {val_root}")
    pin_memory = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate,
    )
    return train_loader, val_loader
