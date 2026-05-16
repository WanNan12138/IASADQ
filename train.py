import torch

from config import parse_args
from data import build_dataloaders
from engine.trainer import ISSAQTrainer
from losses import JointQuantLoss
from models import build_backbone
from quant import build_issaq_model
from utils import build_logger, load_checkpoint, save_config, set_seed


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = build_logger(args.output_dir, "train")
    save_config(args, args.output_dir)
    logger.info("|===>Total Bit-FLOPs(G): %.2f", args.target_bitops)
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(args)

    logger.info("Building backbone...")
    backbone, task = build_backbone(args)
    logger.info("Wrapping backbone with ISSAQ quantization...")
    model = build_issaq_model(backbone, task, args).to(args.device)

    criterion = JointQuantLoss(
        task=task,
        label_smoothing=args.label_smoothing,
        lambda_bitops=args.lambda_bitops,
        target_bitops=args.target_bitops,
        sharpness_lambda=args.sharpness_lambda,
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, logger)

    trainer = ISSAQTrainer(
        args=args,
        model=model,
        task=task,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        start_epoch=start_epoch,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
