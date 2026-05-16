from contextlib import contextmanager

import torch
import torch.nn.functional as F

from quant.modules import DynamicQuantConv2d, DynamicQuantLinear


def _quant_modules(model):
    return [module for module in model.modules() if isinstance(module, (DynamicQuantConv2d, DynamicQuantLinear))]


def iter_named_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def snapshot_gradients(model):
    gradients = {}
    for name, param in iter_named_trainable_parameters(model):
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()
    return gradients


def assign_gradients(model, gradients):
    for name, param in iter_named_trainable_parameters(model):
        grad = gradients.get(name)
        param.grad = None if grad is None else grad.detach().clone()


def quantization_kl_proxy(fp_outputs, q_outputs, temperature=1.0):
    if not isinstance(fp_outputs, torch.Tensor) or not isinstance(q_outputs, torch.Tensor):
        device = fp_outputs["loss"].device if isinstance(fp_outputs, dict) else q_outputs["loss"].device
        return torch.tensor(0.0, device=device)
    teacher = F.softmax(fp_outputs / temperature, dim=-1)
    student = F.log_softmax(q_outputs / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * (temperature ** 2)


def build_case_perturbations(model, meta, rho_base, rho_gamma, sign_flip_prob):
    modules = _quant_modules(model)
    grads = [module.weight.grad.norm(2) for module in modules if module.weight.grad is not None]
    if not grads:
        return {"independent": {}, "pre_quant": {}, "post_quant": {}}, 0.0
    grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-12
    bit_mean = meta.avg_bits.mean().detach().clamp(min=1.0)
    complexity = meta.complexity.mean().detach().clamp(min=0.0)
    max_bit = max(modules[0].candidate_bits) if modules else 8
    rho = rho_base * ((float(max_bit) / bit_mean.item()) ** rho_gamma) * (1.0 + complexity.item())

    cases = {"independent": {}, "pre_quant": {}, "post_quant": {}}
    for module in modules:
        if module.weight.grad is None:
            continue
        grad = module.weight.grad.detach()
        random_mask = torch.bernoulli(torch.full_like(grad, sign_flip_prob))
        signed_grad = grad * (2.0 * random_mask - 1.0)
        delta = rho * signed_grad / grad_norm
        noise_scale = 1.0 + float(module.last_quant_error.detach().clamp(min=0.0).item())
        cases["independent"][module] = delta
        cases["pre_quant"][module] = delta
        cases["post_quant"][module] = delta * noise_scale
    return cases, rho


class AdaptiveSharpnessScheduler:
    def __init__(self, low_threshold=0.02, high_threshold=0.08, beta=0.4, ema=0.9, alpha=0.5):
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.beta = float(beta)
        self.ema = float(ema)
        self.alpha = float(alpha)
        self.ema_kl = None

    def step(self, kl_value):
        if isinstance(kl_value, torch.Tensor):
            device = kl_value.device
            current = float(kl_value.detach().item())
        else:
            device = "cpu"
            current = float(kl_value)
        if self.ema_kl is None:
            self.ema_kl = current
        else:
            self.ema_kl = self.ema * self.ema_kl + (1.0 - self.ema) * current

        sensitivities = torch.tensor(
            [0.5 * self.alpha, 1.0 * self.alpha, 1.5 * self.alpha],
            device=device,
            dtype=torch.float32,
        )
        soft = torch.softmax(sensitivities * self.ema_kl, dim=0)
        if self.ema_kl <= self.low_threshold:
            hard = torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.ema_kl <= self.high_threshold:
            hard = torch.tensor([0.0, 1.0, 0.0], device=device)
        else:
            hard = torch.tensor([0.0, 0.0, 1.0], device=device)
        weights = (1.0 - self.beta) * soft + self.beta * hard
        return weights / weights.sum().clamp(min=1e-6)


def determine_sharpness_interval(meta, base_cycle, high_bit_cycle, high_bit_threshold):
    avg_bit = float(meta.avg_bits.mean().detach().item())
    complexity_var = float(meta.complexity_var.mean().detach().item())
    if avg_bit >= high_bit_threshold:
        return max(1, int(high_bit_cycle))
    if complexity_var > 0.1:
        return 1
    return max(1, int(base_cycle))


def combine_gradients(vanilla_grads, sharp_grads, orthogonal_buffers, sharpness_lambda, reuse_gamma):
    combined = {}
    new_buffers = {}
    cosine_values = []
    orthogonal_norm = 0.0
    for name, g in vanilla_grads.items():
        sharp = sharp_grads.get(name)
        if sharp is None:
            reuse = orthogonal_buffers.get(name)
            sharp_hat = g if reuse is None else g + reuse_gamma * reuse
            combined[name] = g + sharpness_lambda * sharp_hat
            continue
        g_flat = g.reshape(-1)
        sharp_flat = sharp.reshape(-1)
        denom = g_flat.dot(g_flat).clamp(min=1e-12)
        gp = (sharp_flat.dot(g_flat) / denom) * g
        gv = sharp - gp
        new_buffers[name] = gv.detach().clone()
        cosine = F.cosine_similarity(g_flat.unsqueeze(0), sharp_flat.unsqueeze(0), dim=1).item()
        cosine_values.append(cosine)
        orthogonal_norm += float(gv.norm().item())
        combined[name] = g + sharpness_lambda * sharp
    stats = {
        "cosine": sum(cosine_values) / max(1, len(cosine_values)),
        "orthogonal_norm": orthogonal_norm,
    }
    return combined, new_buffers, stats


def reuse_gradients(vanilla_grads, orthogonal_buffers, sharpness_lambda, reuse_gamma):
    combined = {}
    reuse_norm = 0.0
    for name, g in vanilla_grads.items():
        reuse = orthogonal_buffers.get(name)
        sharp_hat = g if reuse is None else g + reuse_gamma * reuse
        if reuse is not None:
            reuse_norm += float(reuse.norm().item())
        combined[name] = g + sharpness_lambda * sharp_hat
    return combined, {"cosine": 1.0, "orthogonal_norm": reuse_norm}


@contextmanager
def temporary_perturbation(model, perturbations, mode):
    model.clear_perturbations()
    model.set_perturbations(perturbations, mode)
    try:
        yield
    finally:
        model.clear_perturbations()
