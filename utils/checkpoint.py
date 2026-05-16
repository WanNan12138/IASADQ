from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, best_metric=None):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, logger=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint.get("epoch", 0)
    if logger is not None:
        logger.info("Loaded checkpoint from %s at epoch %s", path, epoch)
    return epoch
