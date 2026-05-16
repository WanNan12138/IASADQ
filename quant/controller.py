import math

import torch
import torch.nn.functional as F
from torch import nn


class InstanceAwareBitController(nn.Module):
    def __init__(
        self,
        num_layers,
        candidate_bits,
        in_channels=None,
        hidden_channels=32,
        hidden_dim=128,
        temperature=5.0,
        tau_min=0.5,
        anneal=0.965,
        freeze_stage1=0.5,
        freeze_stage2=0.8,
        patience=3,
        min_improvement=1e-4,
        temperature_boost=0.25,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.candidate_bits = tuple(candidate_bits)
        self.temperature = float(temperature)
        self.initial_temperature = float(temperature)
        self.tau_min = float(tau_min)
        self.tau_max = float(temperature) * 2.0
        self.anneal = float(anneal)
        self.freeze_stage1 = float(freeze_stage1)
        self.freeze_stage2 = float(freeze_stage2)
        self.stage1_frozen = False
        self.stage2_frozen = False
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self.temperature_boost = float(temperature_boost)
        self.best_val_loss = None
        self.bad_epochs = 0

        conv_in_channels = int(in_channels) if in_channels is not None else hidden_channels
        linear_in_features = hidden_channels * 2
        self.proj = nn.Sequential(
            nn.Conv2d(conv_in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_layers * len(self.candidate_bits)),
        )

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _complexity_stats(self, features):
        energy = features.pow(2).mean(dim=(-2, -1))
        probs = energy / energy.sum(dim=1, keepdim=True).clamp(min=1e-6)
        entropy = -(probs * probs.clamp(min=1e-6).log()).sum(dim=1, keepdim=True)
        entropy = entropy / max(1.0, math.log(probs.shape[1] + 1.0))
        variance = features.var(dim=(-2, -1), unbiased=False).mean(dim=1, keepdim=True)
        complexity = entropy + variance
        return complexity, variance, entropy

    def set_epoch(self, epoch, max_epochs):
        self.temperature = max(self.tau_min, self.initial_temperature * (self.anneal ** epoch))
        progress = float(epoch) / max(1, max_epochs - 1)
        if progress >= self.freeze_stage1 and not self.stage1_frozen:
            self._freeze_module(self.proj[:3])
            self.stage1_frozen = True
        if progress >= self.freeze_stage2 and not self.stage2_frozen:
            self._freeze_module(self.proj[3:6])
            self.stage2_frozen = True

    def update_validation_loss(self, val_loss):
        current = float(val_loss)
        if self.best_val_loss is None or current < (self.best_val_loss - self.min_improvement):
            self.best_val_loss = current
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.temperature = min(self.tau_max, self.temperature + self.temperature_boost)
            self.bad_epochs = 0

    def forward(self, features, hard=True):
        pooled = self.proj(features)
        complexity, complexity_var, controller_entropy = self._complexity_stats(features)
        logits_input = pooled.flatten(1) + complexity
        logits = self.mlp(logits_input)
        logits = logits.view(features.shape[0], self.num_layers, len(self.candidate_bits))
        if self.training:
            assignments = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            indices = probs.argmax(dim=-1)
            assignments = torch.zeros_like(probs).scatter_(-1, indices.unsqueeze(-1), 1.0)
        return assignments, logits, {
            "complexity": complexity.squeeze(-1),
            "complexity_var": complexity_var.squeeze(-1),
            "controller_entropy": controller_entropy.squeeze(-1),
        }
