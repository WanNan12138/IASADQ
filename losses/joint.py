import torch
from torch import nn


class JointQuantLoss(nn.Module):
    def __init__(
        self,
        task="classification",
        label_smoothing=0.0,
        lambda_bitops=1e-4,
        target_bitops=0.0,
        sharpness_lambda=0.5,
    ):
        super().__init__()
        self.task = task
        self.lambda_bitops = lambda_bitops
        self.target_bitops = float(target_bitops)
        self.sharpness_lambda = float(sharpness_lambda)
        self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def task_loss(self, outputs, targets):
        if self.task == "detection":
            return outputs["loss"]
        return self.classification_criterion(outputs, targets)

    def bitops_loss(self, bitops):
        if self.target_bitops <= 0:
            return bitops.new_zeros(())
        return torch.relu(bitops - bitops.new_tensor(self.target_bitops))

    def aggregate_sharpness(self, sharpness_losses, weights):
        total = weights[0].new_zeros(())
        for weight, sharpness_loss in zip(weights, sharpness_losses):
            total = total + weight * sharpness_loss
        return total

    def combine(self, task_loss, bitops_loss, sharpness_loss):
        return task_loss + self.lambda_bitops * bitops_loss + self.sharpness_lambda * sharpness_loss
