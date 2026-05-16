from config import parse_args
from engine.evaluator import benchmark_model
from models import build_backbone
from quant import build_issaq_model
from utils import build_logger, load_checkpoint, set_seed


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = build_logger(args.output_dir, "benchmark")
    backbone, task = build_backbone(args)
    model = build_issaq_model(backbone, task, args).to(args.device)
    if args.resume:
        load_checkpoint(args.resume, model, logger=logger)
    benchmark_model(args, model, task, logger)


if __name__ == "__main__":
    main()
