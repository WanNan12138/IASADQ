from config import parse_args
from data import build_dataloaders
from engine.evaluator import evaluate_model
from losses import JointQuantLoss
from models import build_backbone
from quant import build_issaq_model
from utils import build_logger, load_checkpoint, set_seed


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = build_logger(args.output_dir, "eval")
    logger.info("|===>Total Bit-FLOPs(G): %.2f", args.target_bitops)
    _, val_loader = build_dataloaders(args)
    backbone, task = build_backbone(args)
    model = build_issaq_model(backbone, task, args).to(args.device)
    criterion = JointQuantLoss(
        task=task,
        label_smoothing=args.label_smoothing,
        lambda_bitops=args.lambda_bitops,
        target_bitops=args.target_bitops,
        sharpness_lambda=args.sharpness_lambda,
    )
    if args.resume:
        load_checkpoint(args.resume, model, logger=logger)
    evaluate_model(args, model, task, val_loader, criterion, logger)


if __name__ == "__main__":
    main()
