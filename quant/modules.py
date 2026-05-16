import math

import torch
import torch.nn.functional as F
from torch import nn

from quant.fake_quant import quantization_error, symmetric_quantize


class _DynamicQuantBase:
    def _init_quant_state(self, candidate_bits):
        self.candidate_bits = tuple(int(bit) for bit in candidate_bits)
        self.policy = None
        self.quant_enabled = True
        self.perturbation = None
        self.perturbation_mode = None
        self.last_bitops = torch.tensor(0.0)
        self.last_quant_error = torch.tensor(0.0)
        self.last_policy_mass = torch.tensor(0.0)
        self.layer_name = ""

    def set_policy(self, policy):
        self.policy = policy

    def set_quantization_enabled(self, enabled):
        self.quant_enabled = enabled

    def set_perturbation(self, perturbation, mode):
        self.perturbation = perturbation
        self.perturbation_mode = mode

    def clear_perturbation(self):
        self.perturbation = None
        self.perturbation_mode = None

    def _default_policy(self, batch_size, device):
        policy = torch.zeros(batch_size, len(self.candidate_bits), device=device)
        policy[:, len(self.candidate_bits) // 2] = 1.0
        return policy

    def resolved_policy(self, batch_size, device):
        if self.policy is None:
            return self._default_policy(batch_size, device)
        return self.policy.to(device)

    def expected_bitwidths(self, batch_size, device):
        weights = torch.tensor(self.candidate_bits, dtype=torch.float32, device=device)
        policy = self.resolved_policy(batch_size, device)
        return torch.matmul(policy, weights)


class DynamicQuantConv2d(nn.Conv2d, _DynamicQuantBase):
    def __init__(self, *args, candidate_bits, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant_state(candidate_bits)
        self.weight_clip = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.activation_clip = nn.Parameter(torch.tensor(1.0))

    @classmethod
    def from_float(cls, module, candidate_bits):
        quantized = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            candidate_bits=candidate_bits,
        )
        quantized.weight.data.copy_(module.weight.data)
        if module.bias is not None and quantized.bias is not None:
            quantized.bias.data.copy_(module.bias.data)
        per_channel = module.weight.detach().abs().mean(dim=(1, 2, 3), keepdim=True) * 3.0 + 1e-4
        quantized.weight_clip.data.copy_(per_channel)
        quantized.activation_clip.data.fill_(3.0)
        return quantized

    def _bitops(self, output, expected_bits):
        kh, kw = self.kernel_size
        out_h, out_w = output.shape[-2:]
        macs = out_h * out_w * self.out_channels * (self.in_channels / self.groups) * kh * kw
        return output.new_tensor(macs / 1e9) * expected_bits.mean() * expected_bits.mean()

    def _quantized_weight(self, bits):
        weight = self.weight
        if self.perturbation is not None and self.perturbation_mode == "pre_quant":
            weight = weight + self.perturbation
        quantized = symmetric_quantize(weight, bits, self.weight_clip)
        if self.perturbation is not None and self.perturbation_mode == "post_quant":
            quantized = quantized + self.perturbation
        return quantized

    def forward(self, inputs):
        if not self.quant_enabled:
            return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        batch_size = inputs.shape[0]
        policy = self.resolved_policy(batch_size, inputs.device)
        expected_bits = self.expected_bitwidths(batch_size, inputs.device)
        output = None
        error_estimate = inputs.new_zeros(())
        for index, bits in enumerate(self.candidate_bits):
            quantized_input = symmetric_quantize(inputs, bits, self.activation_clip)
            quantized_weight = self._quantized_weight(bits)
            candidate = F.conv2d(
                quantized_input,
                quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            if self.perturbation is not None and self.perturbation_mode == "independent":
                candidate = candidate + F.conv2d(
                    quantized_input,
                    self.perturbation,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            weight = policy[:, index].view(-1, 1, 1, 1)
            output = candidate * weight if output is None else output + candidate * weight
            error_estimate = error_estimate + policy[:, index].mean() * quantization_error(self.weight, quantized_weight).abs().mean()
        self.last_policy_mass = policy.mean(dim=0).detach()
        self.last_quant_error = error_estimate
        self.last_bitops = self._bitops(output, expected_bits)
        return output


class DynamicQuantLinear(nn.Linear, _DynamicQuantBase):
    def __init__(self, *args, candidate_bits, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant_state(candidate_bits)
        self.weight_clip = nn.Parameter(torch.ones(self.out_features, 1))
        self.activation_clip = nn.Parameter(torch.tensor(1.0))

    @classmethod
    def from_float(cls, module, candidate_bits):
        quantized = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            candidate_bits=candidate_bits,
        )
        quantized.weight.data.copy_(module.weight.data)
        if module.bias is not None and quantized.bias is not None:
            quantized.bias.data.copy_(module.bias.data)
        per_channel = module.weight.detach().abs().mean(dim=1, keepdim=True) * 3.0 + 1e-4
        quantized.weight_clip.data.copy_(per_channel)
        quantized.activation_clip.data.fill_(3.0)
        return quantized

    def _bitops(self, output, expected_bits):
        macs = self.in_features * self.out_features
        return output.new_tensor(macs / 1e9) * expected_bits.mean() * expected_bits.mean()

    def _quantized_weight(self, bits):
        weight = self.weight
        if self.perturbation is not None and self.perturbation_mode == "pre_quant":
            weight = weight + self.perturbation
        quantized = symmetric_quantize(weight, bits, self.weight_clip)
        if self.perturbation is not None and self.perturbation_mode == "post_quant":
            quantized = quantized + self.perturbation
        return quantized

    def forward(self, inputs):
        if not self.quant_enabled:
            return F.linear(inputs, self.weight, self.bias)
        batch_size = inputs.shape[0]
        policy = self.resolved_policy(batch_size, inputs.device)
        expected_bits = self.expected_bitwidths(batch_size, inputs.device)
        output = None
        error_estimate = inputs.new_zeros(())
        for index, bits in enumerate(self.candidate_bits):
            quantized_input = symmetric_quantize(inputs, bits, self.activation_clip)
            quantized_weight = self._quantized_weight(bits)
            candidate = F.linear(quantized_input, quantized_weight, self.bias)
            if self.perturbation is not None and self.perturbation_mode == "independent":
                candidate = candidate + F.linear(quantized_input, self.perturbation, None)
            weight = policy[:, index].view(-1, 1)
            output = candidate * weight if output is None else output + candidate * weight
            error_estimate = error_estimate + policy[:, index].mean() * quantization_error(self.weight, quantized_weight).abs().mean()
        self.last_policy_mass = policy.mean(dim=0).detach()
        self.last_quant_error = error_estimate
        self.last_bitops = self._bitops(output, expected_bits)
        return output
