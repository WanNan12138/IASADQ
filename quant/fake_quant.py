import torch


def ste_round(x):
    return (x.round() - x).detach() + x


def symmetric_quantize(x, bits, clip_value):
    if bits >= 32:
        return x
    clip_value = clip_value.abs().clamp(min=1e-6)
    x = x.clamp(-clip_value, clip_value)
    if bits <= 1:
        return torch.where(x >= 0, clip_value, -clip_value)
    qmax = max(1, 2 ** (bits - 1) - 1)
    scale = qmax / clip_value
    q = ste_round(x * scale).clamp(-qmax - 1, qmax)
    return q / scale


def quantization_error(reference, quantized):
    return quantized - reference
