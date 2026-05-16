import time

import torch

from utils import AverageMeter, accuracy


@torch.no_grad()
def evaluate_model(args, model, task, loader, criterion, logger):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    bitops_meter = AverageMeter()
    bits_meter = AverageMeter()

    for step, batch in enumerate(loader):
        if args.max_val_steps > 0 and step >= args.max_val_steps:
            break
        images, targets = _move_batch(batch, args.device, task)
        outputs, meta = model(images, targets=targets, hard=True)
        loss = criterion.task_loss(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        bitops_meter.update(meta.bitops.detach().item(), images.size(0))
        bits_meter.update(meta.avg_bits.mean().detach().item(), images.size(0))
        if task == "classification":
            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            top1_meter.update(top1.item(), images.size(0))
            top5_meter.update(top5.item(), images.size(0))
        if step % max(1, args.print_freq) == 0:
            progress = _format_progress(step, len(loader))
            if task == "classification":
                logger.info(
                    "Test: %s  Loss %.4f  ACC1 %.2f%%  ACC5 %.2f%%  Bits %.2f",
                    progress,
                    loss_meter.avg,
                    top1_meter.avg,
                    top5_meter.avg,
                    bits_meter.avg,
                )
            else:
                logger.info(
                    "Test: %s  Loss %.4f  Bits %.2f",
                    progress,
                    loss_meter.avg,
                    bits_meter.avg,
                )

    if task == "classification":
        logger.info(
            "Validation: Loss %.4f  ACC1 %.2f%%  ACC5 %.2f%%  Bit-FLOPs %.2fG",
            loss_meter.avg,
            top1_meter.avg,
            top5_meter.avg,
            bitops_meter.avg,
        )
        return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg, "bitops": bitops_meter.avg}
    logger.info("Validation: Loss %.4f  Bit-FLOPs %.2fG", loss_meter.avg, bitops_meter.avg)
    return {"loss": loss_meter.avg, "bitops": bitops_meter.avg}


@torch.no_grad()
def benchmark_model(args, model, task, logger):
    model.eval()
    device = args.device
    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    for _ in range(args.benchmark_warmup):
        _ = model(dummy, hard=True) if task == "classification" else model(dummy, targets=None, hard=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.benchmark_iters):
        _ = model(dummy, hard=True) if task == "classification" else model(dummy, targets=None, hard=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    samples = args.batch_size * args.benchmark_iters
    throughput = samples / elapsed
    latency_ms = elapsed * 1000.0 / args.benchmark_iters
    logger.info("Benchmark | latency %.3f ms/iter | throughput %.2f samples/s", latency_ms, throughput)
    return {"latency_ms": latency_ms, "throughput": throughput}


def _move_batch(batch, device, task):
    images, targets = batch
    images = images.to(device)
    if task == "classification":
        return images, targets.to(device)
    targets["yolo_targets"] = targets["yolo_targets"].to(device)
    return images, targets


def _format_progress(step, total_steps):
    return f"[{step:4d}/{total_steps}]"
