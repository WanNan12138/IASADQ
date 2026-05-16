from torch import nn

from models.feature_wrappers import TorchvisionMobileNetV2Backbone


def mobilenetv2_imagenet(
    num_classes: int = 1000,
    pretrained: bool = False,
    width: float = 1.0,
    **_: object,
) -> nn.Module:
    return TorchvisionMobileNetV2Backbone(num_classes=num_classes, pretrained=pretrained, width=width)
