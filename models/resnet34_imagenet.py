from torch import nn

from models.feature_wrappers import TorchvisionResNetBackbone


def resnet34_imagenet(
    num_classes: int = 1000,
    pretrained: bool = False,
    width: float = 1.0,
    **_: object,
) -> nn.Module:
    if width != 1.0:
        raise ValueError("resnet34_imagenet currently supports width=1.0 only.")
    return TorchvisionResNetBackbone(
        "resnet34",
        num_classes=num_classes,
        pretrained=pretrained,
        controller_feature_channels=64,
    )
