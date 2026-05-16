import argparse
from datetime import datetime
from pathlib import Path


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_bits(value):
    if value in (None, ""):
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(int(float(v)) for v in value)
    return tuple(int(float(v.strip())) for v in str(value).split(",") if v.strip())


def _build_default_candidate_bits(target_bit):
    target = int(round(float(target_bit)))
    if target <= 2:
        return (1, 2, 3)
    if target == 3:
        return (1, 2, 3, 4, 5)
    candidates = []
    for bit in range(target - 2, target + 3):
        if bit >= 1:
            candidates.append(bit)
    return tuple(sorted(set(candidates)))


def _get_parser():
    parser = argparse.ArgumentParser("ISSAQ training and evaluation")

    # DynamicQuant-style positional dataset root.
    parser.add_argument("data", nargs="?", default="", help="path to dataset root")

    # Common task arguments.
    parser.add_argument("--task", default="classification", choices=["classification", "detection"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet", "yolo"])
    parser.add_argument("--data-dir", default="", help="dataset root for classification tasks")
    parser.add_argument("--det-data", default="", help="YOLO-style data yaml for detection tasks")
    parser.add_argument("-a", "--arch", default="resnet18")
    parser.add_argument("--network", default="", help="SAQ-style alias for --arch")
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--n_classes", default=None, type=int, help="SAQ-style alias for --num-classes")
    parser.add_argument("--image-size", default=224, type=int)

    # Original train.py style optimization args.
    parser.add_argument("--epochs", "--n_epochs", dest="epochs", default=120, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--opt_type", default="", help="SAQ-style optimizer name")
    parser.add_argument("--lr", "--learning-rate", dest="lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--label_smooth", dest="label_smoothing_alias", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--pretrained", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", default=5.0, type=float)
    parser.add_argument("--max-train-steps", default=0, type=int, help="limit train steps per epoch for smoke tests")
    parser.add_argument("--max-val-steps", default=0, type=int, help="limit eval steps for smoke tests")

    # Logging / save path aliases close to original projects.
    parser.add_argument("--output-dir", default="", help="ISSAQ output directory")
    parser.add_argument("--save_dir", default="", help="DynamicQuant-style alias for output directory")
    parser.add_argument("--save_path", default="", help="SAQ-style alias for output directory")
    parser.add_argument("--suffix", default="", help="optional experiment suffix")
    parser.add_argument("--experiment_id", default="exp")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))

    # Quantization args from SAQ + DynamicQuant.
    parser.add_argument("--bits", default="", help="candidate bits as comma-separated values")
    parser.add_argument("--bits_choice", default="", help="SAQ-style alias for candidate bits")
    parser.add_argument("--arch_bits", default="", help="fixed layer bit configuration")
    parser.add_argument("--quantize-first-last", "--quantize_first_last", dest="quantize_first_last", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("--quan_type", default="LIQ_wn")
    parser.add_argument("--quant_method", default="dorefa")
    parser.add_argument("--qw", default=4.0, type=float)
    parser.add_argument("--qa", default=4.0, type=float)
    parser.add_argument("--clip_lr", default=0.01, type=float)
    parser.add_argument("--share_clip", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("--wa_same_bit", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--search_w_bit", nargs="?", const=True, default=False, type=_str2bool)

    # Controller / DynamicQuant args.
    parser.add_argument("--controller-hidden", default=128, type=int)
    parser.add_argument("--hidden_size", default=None, type=int, help="SAQ controller hidden size alias")
    parser.add_argument("--controller-channels", default=32, type=int)
    parser.add_argument("--controller-temperature", default=5.0, type=float)
    parser.add_argument("--controller-tau-min", default=0.5, type=float)
    parser.add_argument("--controller-anneal", default=0.965, type=float)
    parser.add_argument("--freeze-controller-stage1", default=0.5, type=float)
    parser.add_argument("--freeze-controller-stage2", default=0.8, type=float)
    parser.add_argument("--controller-patience", default=3, type=int)
    parser.add_argument("--controller-min-improve", default=1e-4, type=float)
    parser.add_argument("--controller-temp-boost", default=0.25, type=float)
    parser.add_argument("--controller-hard", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--c_lr", default=0.001, type=float)
    parser.add_argument("--c_n_epochs", default=90, type=int)
    parser.add_argument("--c_weight_decay", default=5e-4, type=float)
    parser.add_argument("--c_pretrained", default="")
    parser.add_argument("--c_resume", default="")
    parser.add_argument("--train_ratio", default=0.5, type=float)
    parser.add_argument("--val_num", default=10000, type=int)
    parser.add_argument("--entropy_coeff", default=5e-4, type=float)
    parser.add_argument("--bit_warmup_epochs", default=0, type=int)

    # Sharpness-aware args.
    parser.add_argument("--rho-base", "--rho", dest="rho_base", default=0.05, type=float)
    parser.add_argument("--rho-gamma", default=1.0, type=float)
    parser.add_argument("--eta", default=0.01, type=float)
    parser.add_argument("--include_wclip", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("--include_aclip", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--include_bn", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--sign-flip-prob", default=0.15, type=float)
    parser.add_argument("--sharpness-lambda", default=0.5, type=float)
    parser.add_argument("--reuse-gamma", default=0.5, type=float)
    parser.add_argument("--sharpness-cycle", default=2, type=int)
    parser.add_argument("--high-bit-threshold", default=6.0, type=float)
    parser.add_argument("--high-bit-cycle", default=4, type=int)

    # Joint loss / scheduler args.
    parser.add_argument("--lambda-bitops", default=1e-4, type=float)
    parser.add_argument("--loss_lambda", default=None, type=float, help="SAQ-style alias for lambda-bitops")
    parser.add_argument("--target-bitops", default=0.0, type=float)
    parser.add_argument("--target_bops", default=None, type=float, help="SAQ-style alias for target-bitops")
    parser.add_argument("--tar_bit", "--tb", dest="tar_bit", default=4.0, type=float)
    parser.add_argument("--alpha", "--al", dest="alpha", default=0.05, type=float)
    parser.add_argument("--scheduler-ema", default=0.9, type=float)
    parser.add_argument("--scheduler-beta", default=0.4, type=float)
    parser.add_argument("--scheduler-low", default=0.02, type=float)
    parser.add_argument("--scheduler-high", default=0.08, type=float)

    # Misc compatibility args from original train.py files.
    parser.add_argument("--width", default=1.0, type=float)
    parser.add_argument("-r", default=1.0, type=float, dest="r")
    parser.add_argument("--lr_scheduler", default="Cosine", type=str)
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--multiprocessing-distributed", action="store_true")
    parser.add_argument("--init_method", default="")
    parser.add_argument("--benchmark-iters", default=100, type=int)
    parser.add_argument("--benchmark-warmup", default=20, type=int)

    # YOLOv5 adapter args.
    parser.add_argument("--yolov5-repo", default="", help="local path to YOLOv5 repo")
    parser.add_argument("--yolov5-cfg", default="", help="YOLOv5 model yaml/cfg")
    parser.add_argument("--yolov5-weights", default="", help="YOLOv5 checkpoint path")
    return parser


def _finalize_args(args):
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.network:
        args.arch = args.network
    if args.n_classes is not None:
        args.num_classes = args.n_classes
    if args.label_smoothing_alias is not None:
        args.label_smoothing = args.label_smoothing_alias
    if args.hidden_size is not None:
        args.controller_hidden = args.hidden_size
    if args.loss_lambda is not None:
        args.lambda_bitops = args.loss_lambda
    if args.target_bops is not None:
        args.target_bitops = args.target_bops

    if args.opt_type:
        opt_name = args.opt_type.lower()
        if "adamw" in opt_name:
            args.optimizer = "adamw"
        else:
            args.optimizer = "sgd"

    output_root = args.output_dir or args.save_dir or args.save_path
    if not output_root:
        output_root = str(Path("runs") / "issaq")
    output_root = Path(output_root)
    if args.suffix:
        output_root = output_root / args.suffix
    args.output_dir = str(output_root)

    arch_bits = _parse_bits(args.arch_bits)
    args.arch_bits = arch_bits

    candidate_bits = ()
    if args.bits:
        candidate_bits = _parse_bits(args.bits)
    elif args.bits_choice:
        candidate_bits = _parse_bits(args.bits_choice)
    else:
        candidate_bits = _build_default_candidate_bits(args.tar_bit)
    if not candidate_bits:
        candidate_bits = _build_default_candidate_bits(args.tar_bit)
    args.bits = tuple(sorted(set(candidate_bits)))
    args.bits_choice = args.bits

    if args.dataset == "imagenet" and args.num_classes == 10:
        args.num_classes = 1000
    if args.dataset == "cifar100" and args.num_classes == 10:
        args.num_classes = 100
    if args.arch.startswith("yolov5") or args.task == "detection":
        args.task = "detection"
        args.dataset = "yolo"

    return args


def parse_args(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)
    return _finalize_args(args)
