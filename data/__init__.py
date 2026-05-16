from data.classification import build_classification_loaders
from data.detection import build_detection_loaders


def build_dataloaders(args):
    if args.task == "detection":
        return build_detection_loaders(args)
    return build_classification_loaders(args)
