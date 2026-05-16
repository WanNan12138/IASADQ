from dataclasses import dataclass

import torch
from torch import nn

from quant.controller import InstanceAwareBitController
from quant.modules import DynamicQuantConv2d, DynamicQuantLinear


@dataclass
class QuantizationMeta:
    bitops: torch.Tensor
    avg_bits: torch.Tensor
    layer_bits: torch.Tensor
    assignments: torch.Tensor
    logits: torch.Tensor
    temperature: float
    complexity: torch.Tensor
    complexity_var: torch.Tensor
    controller_entropy: torch.Tensor
    quant_error: torch.Tensor
    bit_usage: torch.Tensor


def _resolve_parent(model, name):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _replace_quant_modules(model, candidate_bits, quantize_first_last=False):
    named = [(name, module) for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    skip = set()
    if named and not quantize_first_last:
        skip.add(named[0][0])
        skip.add(named[-1][0])
    quant_modules = []
    for name, module in named:
        if name in skip:
            continue
        parent, child_name = _resolve_parent(model, name)
        if isinstance(module, nn.Conv2d):
            replacement = DynamicQuantConv2d.from_float(module, candidate_bits)
        else:
            replacement = DynamicQuantLinear.from_float(module, candidate_bits)
        replacement.layer_name = name
        if child_name.isdigit():
            parent[int(child_name)] = replacement
        else:
            setattr(parent, child_name, replacement)
        quant_modules.append(replacement)
    return quant_modules


class ISSAQModel(nn.Module):
    def __init__(self, backbone, task, candidate_bits, controller, quantize_first_last=False):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.candidate_bits = tuple(candidate_bits)
        self.quant_modules = _replace_quant_modules(
            self.backbone,
            candidate_bits=self.candidate_bits,
            quantize_first_last=quantize_first_last,
        )
        self.controller = controller

    def _forward_backbone(self, images, targets=None):
        if self.task == "detection":
            return self.backbone(images, targets)
        return self.backbone(images)

    def _controller_features(self, images):
        if not hasattr(self.backbone, "extract_controller_features"):
            return images
        self.set_quantization_enabled(False)
        try:
            return self.backbone.extract_controller_features(images)
        finally:
            self.set_quantization_enabled(True)

    def _assign_policies(self, assignments):
        for index, module in enumerate(self.quant_modules):
            module.set_policy(assignments[:, index, :])

    def _collect_meta(self, assignments, logits, controller_meta):
        if not self.quant_modules:
            zero = logits.new_zeros(())
            return QuantizationMeta(
                bitops=zero,
                avg_bits=logits.new_zeros(logits.shape[0]),
                layer_bits=logits.new_zeros((logits.shape[0], 0)),
                assignments=assignments,
                logits=logits,
                temperature=getattr(self.controller, "temperature", 1.0),
                complexity=logits.new_zeros(logits.shape[0]),
                complexity_var=logits.new_zeros(logits.shape[0]),
                controller_entropy=logits.new_zeros(logits.shape[0]),
                quant_error=zero,
                bit_usage=logits.new_zeros((0, 0)),
            )
        layer_bits = torch.stack(
            [module.expected_bitwidths(assignments.shape[0], assignments.device) for module in self.quant_modules],
            dim=1,
        )
        bitops = torch.stack([module.last_bitops.to(assignments.device) for module in self.quant_modules]).sum()
        quant_error = torch.stack([module.last_quant_error.to(assignments.device) for module in self.quant_modules]).mean()
        bit_usage = torch.stack([module.last_policy_mass.to(assignments.device) for module in self.quant_modules], dim=0)
        return QuantizationMeta(
            bitops=bitops,
            avg_bits=layer_bits.mean(dim=1),
            layer_bits=layer_bits,
            assignments=assignments,
            logits=logits,
            temperature=getattr(self.controller, "temperature", 1.0),
            complexity=controller_meta["complexity"],
            complexity_var=controller_meta["complexity_var"],
            controller_entropy=controller_meta["controller_entropy"],
            quant_error=quant_error,
            bit_usage=bit_usage,
        )

    def forward(self, images, targets=None, hard=True):
        if self.quant_modules:
            controller_features = self._controller_features(images)
            assignments, logits, controller_meta = self.controller(controller_features, hard=hard)
            self._assign_policies(assignments)
        else:
            logits = images.new_zeros((images.shape[0], 0, 0))
            assignments = images.new_zeros((images.shape[0], 0, 0))
            controller_meta = {
                "complexity": images.new_zeros(images.shape[0]),
                "complexity_var": images.new_zeros(images.shape[0]),
                "controller_entropy": images.new_zeros(images.shape[0]),
            }
        outputs = self._forward_backbone(images, targets)
        meta = self._collect_meta(assignments, logits, controller_meta)
        return outputs, meta

    def forward_full_precision(self, images, targets=None):
        self.set_quantization_enabled(False)
        try:
            return self._forward_backbone(images, targets)
        finally:
            self.set_quantization_enabled(True)

    def set_quantization_enabled(self, enabled):
        for module in self.quant_modules:
            module.set_quantization_enabled(enabled)

    def clear_perturbations(self):
        for module in self.quant_modules:
            module.clear_perturbation()

    def set_perturbations(self, perturbations, mode):
        for module, perturbation in perturbations.items():
            module.set_perturbation(perturbation, mode)


def build_issaq_model(backbone, task, args):
    total_layers = len([module for module in backbone.modules() if isinstance(module, (nn.Conv2d, nn.Linear))])
    num_layers = max(0, total_layers - (0 if args.quantize_first_last else 2))
    controller = InstanceAwareBitController(
        num_layers=num_layers,
        candidate_bits=args.bits,
        in_channels=getattr(backbone, "controller_feature_channels", 3),
        hidden_channels=args.controller_channels,
        hidden_dim=args.controller_hidden,
        temperature=args.controller_temperature,
        tau_min=args.controller_tau_min,
        anneal=args.controller_anneal,
        freeze_stage1=args.freeze_controller_stage1,
        freeze_stage2=args.freeze_controller_stage2,
        patience=args.controller_patience,
        min_improvement=args.controller_min_improve,
        temperature_boost=args.controller_temp_boost,
    )
    return ISSAQModel(
        backbone=backbone,
        task=task,
        candidate_bits=args.bits,
        controller=controller,
        quantize_first_last=args.quantize_first_last,
    )
