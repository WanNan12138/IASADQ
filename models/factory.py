from models.mobilenetv2_imagenet import mobilenetv2_imagenet
from models.resnet18_imagenet import resnet18_imagenet
from models.resnet20_cifar import resnet20_cifar
from models.resnet34_imagenet import resnet34_imagenet
from models.resnet50_imagenet import resnet50_imagenet
from models.yolov5_adapter import YOLOv5Adapter


MODEL_REGISTRY = {
    "resnet20": resnet20_cifar,
    "resnet20_cifar": resnet20_cifar,
    "resnet18": resnet18_imagenet,
    "resnet18_imagenet": resnet18_imagenet,
    "resnet34": resnet34_imagenet,
    "resnet34_imagenet": resnet34_imagenet,
    "resnet50": resnet50_imagenet,
    "resnet50_imagenet": resnet50_imagenet,
    "mobilenet_v2": mobilenetv2_imagenet,
    "mobilenetv2": mobilenetv2_imagenet,
    "mobilenetv2_imagenet": mobilenetv2_imagenet,
}


def build_backbone(args):
    if args.task == "detection":
        model = YOLOv5Adapter(
            repo_path=args.yolov5_repo,
            cfg=args.yolov5_cfg,
            weights=args.yolov5_weights,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
        return model, "detection"
    if args.arch not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported classification architecture: {args.arch}")
    model = MODEL_REGISTRY[args.arch](
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained),
        width=float(getattr(args, "width", 1.0)),
    )
    return model, "classification"
