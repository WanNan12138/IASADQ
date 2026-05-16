import inspect

import torch
import torchvision.models as tv_models
from torch import nn


def _load_torchvision_model(builder, weight_enum_name, pretrained, **kwargs):
    if "weights" in inspect.signature(builder).parameters:
        weights = getattr(tv_models, weight_enum_name).DEFAULT if pretrained else None
        return builder(weights=weights, **kwargs)
    return builder(pretrained=pretrained, **kwargs)


class TorchvisionResNetBackbone(nn.Module):
    def __init__(self, variant, num_classes=1000, pretrained=False, controller_feature_channels=64):
        super().__init__()
        builder = getattr(tv_models, variant)
        weight_enum_name = f"{variant.capitalize()}_Weights"
        self.model = _load_torchvision_model(builder, weight_enum_name, pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.controller_feature_channels = int(controller_feature_channels)

    def extract_controller_features(self, x):
        model = self.model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        return x

    def forward(self, x):
        return self.model(x)


class TorchvisionMobileNetV2Backbone(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, width=1.0):
        super().__init__()
        self.model = _load_torchvision_model(
            tv_models.mobilenet_v2,
            "MobileNet_V2_Weights",
            pretrained,
            width_mult=width,
        )
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.controller_feature_channels = max(8, int(round(24 * width)))

    def extract_controller_features(self, x):
        for layer in self.model.features[:4]:
            x = layer(x)
        return x

    def forward(self, x):
        return self.model(x)
