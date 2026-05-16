import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _lookup_pickle_key(record, *keys):
    for key in keys:
        if key in record:
            return record[key]
        encoded = key.encode("utf-8")
        if encoded in record:
            return record[encoded]
    raise KeyError(f"Could not find any of keys {keys} in pickle record.")


def _decode_list(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _resolve_cifar_root(root, dataset):
    root = Path(root)
    candidates = [root]
    if dataset == "cifar10":
        candidates.extend([root / "cifar-10-batches-py", root / "cifar-10-python"])
        required = ("data_batch_1", "test_batch", "batches.meta")
    else:
        candidates.extend([root / "cifar-100-python", root / "cifar-100-batches-py"])
        required = ("train", "test", "meta")
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {dataset} pickle directory from {root}. "
        f"Expected files: {', '.join(required)}."
    )


def _resolve_imagenet_root(root):
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet root {root} must contain both 'train' and 'val' directories."
        )
    if not any(train_dir.iterdir()):
        raise FileNotFoundError(f"ImageNet train directory is empty: {train_dir}")
    if not any(val_dir.iterdir()):
        raise FileNotFoundError(f"ImageNet val directory is empty: {val_dir}")
    return train_dir, val_dir


def _classification_transforms(dataset, image_size, train):
    if dataset == "imagenet":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    mean, std = CIFAR_STATS[dataset]
    resize_ops = []
    if image_size != 32:
        resize_ops.append(transforms.Resize((image_size, image_size), antialias=True))
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *resize_ops,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            *resize_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class PickleCIFARDataset(Dataset):
    def __init__(self, root, dataset, train, transform=None):
        self.root = _resolve_cifar_root(root, dataset)
        self.dataset = dataset
        self.train = train
        self.transform = transform
        self.images, self.targets, self.classes = self._load_split()

    def _load_pickle(self, path):
        with open(path, "rb") as handle:
            return pickle.load(handle, encoding="latin1")

    def _load_split(self):
        if self.dataset == "cifar10":
            filenames = [f"data_batch_{idx}" for idx in range(1, 6)] if self.train else ["test_batch"]
            meta = self._load_pickle(self.root / "batches.meta")
            classes = _decode_list(_lookup_pickle_key(meta, "label_names"))
            label_keys = ("labels",)
        else:
            filenames = ["train"] if self.train else ["test"]
            meta = self._load_pickle(self.root / "meta")
            classes = _decode_list(_lookup_pickle_key(meta, "fine_label_names"))
            label_keys = ("fine_labels", "labels")

        data_parts = []
        targets = []
        for filename in filenames:
            record = self._load_pickle(self.root / filename)
            data_parts.append(np.asarray(_lookup_pickle_key(record, "data")))
            labels = _lookup_pickle_key(record, *label_keys)
            targets.extend(int(label) for label in labels)

        images = np.concatenate(data_parts, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images.astype(np.uint8), targets, classes

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _build_cifar_dataset(args, train):
    dataset_name = args.dataset
    image_size = args.image_size if args.image_size > 0 else 32
    return PickleCIFARDataset(
        root=args.data_dir,
        dataset=dataset_name,
        train=train,
        transform=_classification_transforms(dataset_name, image_size, train),
    )


def build_classification_loaders(args):
    root = Path(args.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if args.dataset in {"cifar10", "cifar100"}:
        train_set = _build_cifar_dataset(args, train=True)
        val_set = _build_cifar_dataset(args, train=False)
    else:
        train_dir, val_dir = _resolve_imagenet_root(root)
        train_set = datasets.ImageFolder(
            train_dir,
            transform=_classification_transforms("imagenet", args.image_size, True),
        )
        val_set = datasets.ImageFolder(
            val_dir,
            transform=_classification_transforms("imagenet", args.image_size, False),
        )

    pin_memory = torch.cuda.is_available() and str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
